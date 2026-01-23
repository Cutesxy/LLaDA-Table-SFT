# eval_core/models.py
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel
import logging

logger = logging.getLogger(__name__)

# =======================================================
# 1. LLaDA 核心辅助函数 (Masked Diffusion Helpers)
# =======================================================

def add_gumbel_noise(logits, temperature):
    """
    为 logits 添加 Gumbel 噪声，用于采样
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    """
    计算每一步需要去噪（Transfer）的 Token 数量
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


# =======================================================
# 2. LLaDA 模型封装类
# =======================================================
class LLaDAModelWrapper:
    def __init__(self, model_path, adapter_path=None, steps=64, gen_length=64, block_length=128):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[LLaDA] Loading Base Model from: {model_path}")
        
        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            fix_mistral_regex=True
        )
        if self.tokenizer.padding_side != 'left': 
            self.tokenizer.padding_side = 'left'
        
        # 2. 加载 Base Model
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            dtype=torch.bfloat16 
        ).to(self.device)
        
        # 3. 加载 LoRA Adapter (如果存在)
        if adapter_path:
            logger.info(f"[LLaDA] Loading LoRA Adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
        self.model.eval()
        
        # 4. 设置生成参数
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.mask_id = 126336 # LLaDA 固定的 Mask Token ID

    def _diffusion_generate(self, prompt_ids, attention_mask, temperature=0.0, remasking='low_confidence'):
        """
        LLaDA 核心生成逻辑，包含 Block 划分和重掩码机制
        """
        model = self.model
        gen_length = self.gen_length
        steps = self.steps
        block_length = self.block_length
        mask_id = self.mask_id
        
        # 初始化画布：[Prompt, Mask, Mask, ...]
        x = torch.full((prompt_ids.shape[0], prompt_ids.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt_ids.shape[1]] = prompt_ids.clone()

        # 扩展 Attention Mask
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((prompt_ids.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)
            ], dim=-1)

        # 处理 Block 逻辑
        if gen_length < block_length: block_length = gen_length
        if gen_length % block_length != 0: block_length = gen_length 
        
        num_blocks = gen_length // block_length
        steps_per_block = steps // num_blocks if num_blocks > 0 else steps

        # Block 循环
        for num_block in range(num_blocks):
            start_pos = prompt_ids.shape[1] + num_block * block_length
            end_pos = prompt_ids.shape[1] + (num_block + 1) * block_length
            
            block_mask_index = (x[:, start_pos:end_pos] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            
            # Diffusion Step 循环
            for i in range(steps_per_block):
                mask_index = (x == mask_id)
                
                # 前向传播
                logits = model(x, attention_mask=attention_mask).logits

                # 采样
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # 计算置信度
                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

                x0_p[:, end_pos:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    k = num_transfer_tokens[j, i]
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True
                
                x[transfer_index] = x0[transfer_index]
                
        return x

    def generate(self, messages, max_new_tokens=None):
        """
        修改后的生成接口：
        1. 支持 Raw String 输入（跳过 Chat Template）
        2. 支持 EOS 截断
        """
        # 更新生成长度
        if max_new_tokens is not None:
            self.gen_length = max_new_tokens

        # ================= 改动点 1: 输入处理 =================
        if isinstance(messages, str):
            # [模式 A] 填空模式 (Completion Mode)
            # 直接使用字符串，不加 <|im_start|> 等标签
            text_input = messages
        else:
            # [模式 B] 对话模式 (Chat Mode)
            # 使用官方模板添加角色标签
            text_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Tokenize
        encoded = self.tokenizer(text_input, return_tensors='pt', truncation=True, max_length=4096).to(self.device)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # 执行扩散生成
        with torch.no_grad():
            out_tokens = self._diffusion_generate(input_ids, attention_mask)
        
        # ================= 改动点 2: 解码与截断 =================
        # 只取新生成的部分
        generated_part = out_tokens[:, input_ids.shape[1]:]
        
        # 注意：这里改为 skip_special_tokens=False，因为我们需要看到 EOS token
        raw_pred = self.tokenizer.decode(generated_part[0], skip_special_tokens=False)
        
        clean_pred = raw_pred
        
        # 尝试进行 EOS 截断
        # 1. 优先使用 tokenizer 定义的 eos
        if self.tokenizer.eos_token and self.tokenizer.eos_token in clean_pred:
            clean_pred = clean_pred.split(self.tokenizer.eos_token)[0]
        
        # 2. 兜底清洗：如果没生成 EOS，但生成了 PAD (通常不应该发生，但以防万一)
        if self.tokenizer.pad_token:
            clean_pred = clean_pred.replace(self.tokenizer.pad_token, "")
            
        return clean_pred.strip()


# =======================================================
# 3. HuggingFace 标准模型封装类 (Llama/Qwen)
# =======================================================

class HFModelWrapper:
    def __init__(self, model_path, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[HF] Loading Model from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto" 
        ).eval()
        
        self.terminators = [self.tokenizer.eos_token_id]
        if "<|eot_id|>" in self.tokenizer.all_special_tokens:
            self.terminators.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    def generate(self, messages, max_new_tokens=512):
        text_input = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        inputs = self.tokenizer([text_input], return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, 
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        new_tokens = generated_ids[0][inputs.input_ids.shape[1]:]
        prediction = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return prediction.strip()
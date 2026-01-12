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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.padding_side != 'left': 
            self.tokenizer.padding_side = 'left'
        
        # 2. 加载 Base Model (LLaDA 使用 AutoModel)
        self.model = AutoModel.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        # 3. 加载 LoRA Adapter (如果存在)
        if adapter_path:
            logger.info(f"[LLaDA] Loading LoRA Adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            # self.model.merge_and_unload() # 视显存情况可选择合并，一般保持挂载即可
        
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
        # 步数分配给每个 Block
        steps_per_block = steps // num_blocks if num_blocks > 0 else steps

        # Block 循环
        for num_block in range(num_blocks):
            # 确定当前 Block 的范围
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

                # 计算置信度 (Confidence)
                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    # 获取采样 token 对应的概率
                    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(f"Unknown remasking strategy: {remasking}")

                # 强制忽略尚未生成的后续 Block
                x0_p[:, end_pos:] = -np.inf

                # 更新当前画布
                # x0 是预测值，x 是当前被 mask 的值
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                # 选取置信度最高的 k 个 token 进行 Transfer
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    k = num_transfer_tokens[j, i]
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True
                
                # 更新 x
                x[transfer_index] = x0[transfer_index]
                
        return x

    def generate(self, messages, max_new_tokens=None):
        """
        统一对外接口
        """
        # 如果调用时没指定 max_new_tokens，则使用初始化时的 gen_length
        current_gen_length = max_new_tokens if max_new_tokens is not None else self.gen_length
        # 为了避免逻辑混乱，这里临时更新一下 gen_length，或者确保调用 _diffusion_generate 时传入
        # 简单起见，这里我们暂时信任 self.gen_length 是对齐的，
        # 如果需要动态改变长度，建议重新实例化或修改 _diffusion_generate 接受参数。
        # 为了兼容性，这里我们假设 max_new_tokens 主要用于截断，
        # 但 LLaDA 必须预先分配长度，所以我们强行覆盖 self.gen_length 
        if max_new_tokens is not None:
            self.gen_length = max_new_tokens

        # 1. 应用 Chat Template
        text_input = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # 2. Tokenize
        encoded = self.tokenizer(text_input, return_tensors='pt', truncation=True, max_length=4096).to(self.device)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # 3. 执行扩散生成
        with torch.no_grad():
            out_tokens = self._diffusion_generate(input_ids, attention_mask)
        
        # 4. 解码
        # 只取生成的后面部分
        generated_part = out_tokens[:, input_ids.shape[1]:]
        prediction = self.tokenizer.batch_decode(generated_part, skip_special_tokens=True)[0]
        
        return prediction.strip()


# =======================================================
# 3. HuggingFace 标准模型封装类 (Llama/Qwen)
# =======================================================

class HFModelWrapper:
    def __init__(self, model_path, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[HF] Loading Model from: {model_path}")
        
        # 1. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 2. 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map="auto" # 自动映射到可见 GPU
        ).eval()
        
        # 3. 处理 Llama-3 特有的结束符
        self.terminators = [self.tokenizer.eos_token_id]
        if "<|eot_id|>" in self.tokenizer.all_special_tokens:
            self.terminators.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))

    def generate(self, messages, max_new_tokens=512):
        # 1. 应用 Chat Template
        text_input = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        # 2. Tokenize
        inputs = self.tokenizer([text_input], return_tensors="pt", truncation=True, max_length=4096).to(self.model.device)
        
        # 3. 自回归生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # Greedy Decoding for reproducibility
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 4. 解码
        new_tokens = generated_ids[0][inputs.input_ids.shape[1]:]
        prediction = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return prediction.strip()
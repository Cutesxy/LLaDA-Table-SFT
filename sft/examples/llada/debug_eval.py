import sys
import os
import torch
import torch.nn.functional as F
import logging
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np

# =======================================================
# 1. 基础配置 & 路径
# =======================================================
# [建议] 使用训练后的 Checkpoint 路径
MODEL_PATH = "/home/zjusst/hxy/llada/models/GSAI-ML/LLaDA-8B-Instruct" 
DATA_PATH = "/home/zjusst/hxy/llada/dllm/data/wikitq_test.jsonl" 
NUM_SAMPLES = 435
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mock Scheduler (如果环境里没有 dllm)
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
    from dllm.core.schedulers import LinearAlphaScheduler
    print("[Info] 成功导入 dllm.core.schedulers")
except ImportError:
    print("[Warning] 未找到 dllm 库，使用本地 Mock Scheduler")
    class LinearAlphaScheduler:
        def __call__(self, t): return 1 - t
        def weight(self, t): return 1 / t

# =======================================================
# 2. 核心诊断逻辑 (完全复刻 MDLMTrainer)
# =======================================================
class LossProber:
    def __init__(self, model_path, device):
        self.device = device
        print(f"Loading model from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 1. 修复 Chat Template
        self.tokenizer.chat_template = textwrap.dedent("""\
            {% set loop_messages = messages %}
            {% for message in loop_messages %}
            {% if loop.index0 == 0 %}{{ bos_token }}{% endif %}
            <|start_header_id|>{{ message['role'] }}<|end_header_id|>

            {{ message['content'] | trim }}<|eot_id|>
            {%- endfor %}
            {% if add_generation_prompt and (loop_messages | length == 0 or loop_messages[-1]['role'] != 'assistant') %}
            <|start_header_id|>assistant<|end_header_id|>

            {% endif %}
            """)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16 
        ).to(device).eval()
        
        self.scheduler = LinearAlphaScheduler()
        self.mask_token_id = self.tokenizer.mask_token_id or 126336
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_single_pass_strict(self, input_ids, labels, t_val):
        """
        完全复刻 MDLMTrainer.compute_loss 的逻辑
        """
        b, l = input_ids.shape
        
        # === 1. Sample diffusion timesteps (Trainer Step 1) ===
        # Trainer: t = epsilon + (1-epsilon) * rand
        # Debug: 强制 t 为固定值，但保持 Tensor 形状
        t = torch.tensor([t_val] * b, device=self.device).float()
        
        # Scheduler Logic
        p_mask = 1 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # === 2. Apply stochastic masking (Trainer Step 2) ===
        rand_probs = torch.rand((b, l), device=self.device)
        is_answer = (labels != -100)
        
        masked_indices = (rand_probs < p_mask) & is_answer
        
        # [Debug 特供] 如果是为了测试 t 即使没随机到也强制 mask 一个，方便看 loss
        # 注意：这步会稍微偏离 Trainer (Trainer 会直接 return 0)，但为了诊断是有必要的
        if masked_indices.sum() == 0 and is_answer.sum() > 0:
             first_ans_idx = torch.where(is_answer)[1][0]
             masked_indices[0, first_ans_idx] = True

        noised_input_ids = torch.where(
            masked_indices, 
            torch.tensor(self.mask_token_id, device=self.device), 
            input_ids
        )
        
        # === 3. Forward pass (Trainer Step 3) ===
        with torch.no_grad():
            outputs = self.model(input_ids=noised_input_ids)
            logits = outputs.logits # [B, L, V]

        # === 4. Handle degenerate (Trainer Step 4) ===
        if not masked_indices.any():
            return 0.0, 0.0, 0.0, 0, 0, "No Mask"

        # === 5. Compute weights (Trainer Step 5) ===
        loss_weights = self.scheduler.weight(t).unsqueeze(1).expand(b, l)

        # === 6. Compute loss (Trainer Step 6) ===
        # Flatten for computation
        flat_logits = logits[masked_indices]
        flat_targets = input_ids[masked_indices]
        flat_weights = loss_weights[masked_indices]
        
        # Cross Entropy
        ce_loss = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        token_loss = ce_loss * flat_weights # 乘上时间权重

        # === 7. Normalize (Trainer Step 7 - 关键差异点) ===
        # Trainer 除的是 effective_lengths (答案的总长度)，而不是 masked_indices (被 mask 的数量)
        effective_lengths = torch.sum(labels != -100, dim=1, keepdim=True).expand(b, l)
        
        # 获取每个被 mask token 对应的分母 (对于同一条数据，分母是一样的)
        denoms = effective_lengths[masked_indices]
        
        # Trainer: sum(token_loss / effective_lengths) / b
        final_loss = torch.sum(token_loss / denoms) / b
        
        # === 8. 收集诊断信息 ===
        raw_ce_val = ce_loss.mean().item()
        weight_val = flat_weights[0].item()
        eff_len_val = effective_lengths[0, 0].item() # 答案总长
        mask_cnt_val = masked_indices.sum().item()   # 被Mask的数量

        # 预测检查
        debug_info = {}
        pred_token_id = flat_logits[0].argmax().item()
        true_token_id = flat_targets[0].item()
        debug_info['pred'] = f"'{self.tokenizer.decode([true_token_id])}'->'{self.tokenizer.decode([pred_token_id])}'"
        debug_info['correct'] = (pred_token_id == true_token_id)

        return final_loss, raw_ce_val, weight_val, eff_len_val, mask_cnt_val, debug_info

# =======================================================
# 3. 数据处理 (保持不变，对齐 SFT)
# =======================================================
def default_mdlm_sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    prompt_response_tokens = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False
    )
    labels = prompt_response_tokens.copy()
    if mask_prompt_loss:
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        prompt_len = min(len(prompt_tokens), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {"input_ids": prompt_response_tokens, "labels": labels}
    return {"input_ids": prompt_response_tokens, "labels": labels}

# =======================================================
# 4. 主程序
# =======================================================
def main():
    prober = LossProber(MODEL_PATH, DEVICE)
    
    print(f"Loading dataset: {DATA_PATH}")
    full_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    print(f"随机抽取 {NUM_SAMPLES} 条样本进行测试...")
    dataset = full_dataset.shuffle(seed=42).select(range(NUM_SAMPLES))

    # 表头增加 EffLen (有效长度) 和 MaskCnt (Mask数量)
    header = f"{'ID':<4} | {'Answer (First 15)':<20} | {'t':<6} | {'Loss':<12} | {'Raw CE':<8} | {'Wt':<5} | {'Len/Msk':<8} | {'Pred Check'}"
    print("\n" + "="*len(header))
    print(header)
    print("="*len(header))

    test_t_values = [0.001, 0.01, 0.1]

    for i, row in enumerate(dataset):
        processed = default_mdlm_sft_map_fn(row, tokenizer=prober.tokenizer, mask_prompt_loss=True)
        input_ids = torch.tensor(processed['input_ids']).unsqueeze(0).to(DEVICE)
        labels = torch.tensor(processed['labels']).unsqueeze(0).to(DEVICE)

        valid_label_ids = labels[0][labels[0] != -100]
        if len(valid_label_ids) == 0: continue
        answer_text = prober.tokenizer.decode(valid_label_ids).strip().replace("\n", " ")[:15]
        
        for t in test_t_values:
            # 调用严格复刻版函数
            loss, raw_ce, weight, eff_len, mask_cnt, info = prober.compute_single_pass_strict(input_ids, labels, t)
            
            # 格式化输出
            loss_str = f"{loss:.4f}"
            if loss > 1000: loss_str = f"\033[91m{loss:.2e}\033[0m"
            elif loss > 50: loss_str = f"\033[93m{loss:.4f}\033[0m"
            
            pred_str = info['pred']
            if not info['correct']: pred_str = f"\033[91m{pred_str}\033[0m"
            
            # 显示 总长度/Mask数量
            len_msk_str = f"{eff_len}/{mask_cnt}"
            
            print(f"{i:<4} | {answer_text:<20} | {t:<6.3f} | {loss_str:<12} | {raw_ce:<8.4f} | {weight:<5.0f} | {len_msk_str:<8} | {pred_str}")

        print("-" * len(header))

if __name__ == "__main__":
    main()
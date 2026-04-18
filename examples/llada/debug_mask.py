import sys
import os
import torch
import json
import logging
from functools import partial
import transformers
from transformers import AutoTokenizer 
from datasets import load_dataset
import numpy as np
import textwrap  

# =======================================================
# 1. 路径修复 (让 Python 找到 dllm 库)
# =======================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, project_root)

# 尝试导入 dllm (仅仅为了验证路径，不依赖它的utils了)
try:
    import dllm
    print(f"成功导入 dllm 库 (路径: {project_root})")
except ImportError:
    print(f"警告: 导入 dllm 失败，但这不影响本脚本运行")

# =======================================================
# 2. 用户配置区域
# =======================================================
# 请确认这两个路径是真实存在的
MODEL_PATH = "/home/zjusst/hxy/llada/models/GSAI-ML/LLaDA-8B-Instruct" 
DATA_PATH = "/home/zjusst/hxy/llada/dllm/data/wikitq_test_train.jsonl"
NUM_SAMPLES = 1 

# =======================================================
# 3. 核心函数定义 (直接粘贴进来，不再依赖 import)
# =======================================================
def default_mdlm_sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    """
    Build input_ids and labels for SFT.
    直接定义在这里，避免 AttributeError
    """
    # print("Processing row:")
    # print(row["messages"])
    # 1. 对整个对话进行编码 (User + Assistant)
    prompt_response_tokens = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False
    )
    # =============== [DEBUG START] ===============
    # print("\n" + "="*20 + " 完整对话解码结果 " + "="*20)
    # decode 负责把 ID 变回字符串
    # decoded_str = tokenizer.decode(prompt_response_tokens, skip_special_tokens=False)
    
    # # 打印出来 (repr 可以把换行符显示为 \n，看得更清楚)
    # print(f"Token 总数: {len(prompt_response_tokens)}")
    # print(f"原始内容: \n{decoded_str}")
    # print(f"原始ID: {prompt_response_tokens[-10:]} (只看最后10个)") # 看看最后那个是不是 128009 (eot_id)
    # print("="*60 + "\n")
    # =============== [DEBUG END] =================
    labels = prompt_response_tokens.copy()

    # 2. 如果需要 Mask Prompt Loss (只学习回答部分)
    if mask_prompt_loss:
        # 只编码 User 部分 (作为 Prompt)
        # 注意：这里假设 messages 是标准的 [{'role': 'user'}, {'role': 'assistant'}]
        # [:-1] 取出除了最后一条回复之外的所有内容作为 Prompt
        # print(row["messages"][:-1])
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        
        # 计算长度，确保只 Mask 前面的部分
        prompt_len = len(prompt_tokens)
        
        # 安全检查：如果生成的 Prompt 比整个对话还长 (理论上不应该)，做个截断保护
        if prompt_len > len(labels):
            prompt_len = len(labels)
            
        # 将 Prompt 部分的 Label 设为 -100 (忽略)
        labels[:prompt_len] = [-100] * prompt_len
        
        return {
            "input_ids": prompt_response_tokens,
            "labels": labels,
            "prompt_len": prompt_len,
        }

    return {"input_ids": prompt_response_tokens, "labels": labels}

# =======================================================
# 4. 辅助显示函数
# =======================================================
logging.basicConfig(level=logging.INFO)

def color_print(text, type="normal"):
    RESET = "\033[0m"
    RED = "\033[31m"   # Masked (被遮住的答案)
    GREEN = "\033[32m" # Answer (需要预测的答案)
    BLUE = "\033[34m"  # Prompt (问题部分)
    YELLOW = "\033[33m"# [MASK] 标记
    
    if type == "prompt": return f"{BLUE}{text}{RESET}"
    if type == "answer": return f"{GREEN}{text}{RESET}"
    if type == "masked": return f"{RED}{text}{RESET}"
    if type == "mask_token": return f"{YELLOW}[MASK]{RESET}"
    return text

def simulate_diffusion_mask(input_ids, labels, tokenizer, t_value):
    # 模拟 MDLMTrainer 的 compute_loss 逻辑
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = 126336 # LLaDA 默认保留位
    
    # 根据时间步 t 计算保留概率 p
    # t=1 (噪声大), p=0 (保留少) -> Mask多
    # t=0 (噪声小), p=1 (保留多) -> Mask少
    # 假设线性调度: p_mask = 1 - t
    p_mask = t_value 
    
    is_answer = (labels != -100)
    rand_probs = torch.rand(input_ids.shape)
    
    # 只有是答案(is_answer) 且 随机概率命中(rand < p_mask) 才会被 Mask
    masked_indices = (rand_probs < p_mask) & is_answer
    
    noised_input_ids = input_ids.clone()
    noised_input_ids[masked_indices] = mask_token_id
    
    return noised_input_ids, masked_indices

# =======================================================
# 5. 主程序
# =======================================================
def main():
    print("-" * 60)
    print(f"Loading Tokenizer from: {MODEL_PATH}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Tokenizer 加载失败: {e}")
        return
    
    # 【手动打补丁】覆盖掉有 Bug 的模板 (直接从 dllm 那个文件里抄过来的)
    # 使用 textwrap.dedent 可以自动去除代码缩进带来的多余空格
    tokenizer.chat_template = textwrap.dedent("""\
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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 打印一下 mask token 信息
    print(f"Mask Token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")

    print(f"Loading Dataset from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"找不到数据文件: {DATA_PATH}")
        return
       
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    small_dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))
    # print("原始dataset ------------------------------------")
    # print(small_dataset)
    # print("------------------------------------------------")

    print("\n" + "="*60)
    print("STEP 1: SFT 数据预处理检查")
    print("="*60)
    
    # 使用刚才粘贴进来的本地函数
    map_fn = partial(
        default_mdlm_sft_map_fn,  # <--- 直接调用本地定义的函数
        tokenizer=tokenizer,
        mask_prompt_loss=True, 
    )
    
    processed_dataset = small_dataset.map(map_fn, remove_columns=dataset.column_names)
    
    for i in range(len(processed_dataset)):
        sample = processed_dataset[i]
        # print(sample)
        input_ids = torch.tensor(sample['input_ids'])
        print(input_ids)
        labels = torch.tensor(sample['labels'])
        print(labels)
        
        print(f"\n >>> 样本 {i} (长度: {len(input_ids)}) <<<")
        
        # 静态可视化
        visualized_text = []
        for token_id, label_id in zip(input_ids.tolist(), labels.tolist()):
            token = tokenizer.decode([token_id])
            if label_id == -100:
                visualized_text.append(color_print(token, "prompt"))
            else:
                visualized_text.append(color_print(token, "answer"))
        
        print("".join(visualized_text))
        
        # 动态 Mask 模拟
        print(f"\n--- [动态 Mask 模拟] ---")
        test_timesteps = [0.9, 0.5, 0.1]
        
        for t in test_timesteps:
            noised_ids, masked_indices = simulate_diffusion_mask(
                input_ids, labels, tokenizer, t_value=t
            )
            
            answer_count = (labels != -100).sum().item()
            masked_count = masked_indices.sum().item()
            
            print(f"\n> 时间步 t={t} (期望遮罩率: {1-t:.1f}):")
            print(f"  原本答案长度: {answer_count} -> 被 Mask 掉: {masked_count}")
            
            diff_text = []
            for j, token_id in enumerate(noised_ids.tolist()):
                original_label = labels[j].item()
                if original_label == -100:
                    diff_text.append(color_print(tokenizer.decode([token_id]), "prompt"))
                else:
                    if masked_indices[j]:
                        diff_text.append(color_print("[MASK]", "mask_token"))
                    else:
                        diff_text.append(color_print(tokenizer.decode([token_id]), "answer"))
            
            print("  模型输入: " + "".join(diff_text))
            
            if masked_count == 0 and answer_count > 0:
                print(f"{color_print('警告: 此步没有 Mask 任何内容，Loss 将为 0', 'masked')}")

if __name__ == "__main__":
    main()
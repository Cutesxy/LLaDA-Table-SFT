import transformers
from transformers import AutoTokenizer
import json

# ======================================================
# 1. 设置模型路径 & 加载分词器
# ======================================================
MODEL_PATH = "/home/zjusst/hxy/llada/models/GSAI-ML/LLaDA-8B-Instruct"
print(f"🔄 正在从本地加载分词器: {MODEL_PATH}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ 分词器加载成功！")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    exit()

# ======================================================
# 2. 【关键】直接复制训练代码里的核心函数
#    (这就是“对齐”的核心：保证逻辑代码一个字都不变)
# ======================================================
def default_mdlm_sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    """
    Build input_ids and labels for SFT.
    (这是你 SFT.py 里原本的函数，原封不动复制过来的)
    """
    # 1. 生成完整的 [Prompt + Response]
    prompt_response_tokens = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False
    )
    labels = prompt_response_tokens.copy()

    if mask_prompt_loss:
        # 2. 这里的逻辑：add_generation_prompt=True 
        #    这意味着它会自动计算 User + 固定的 Assistant 头部的长度
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        
        # 3. Masking
        prompt_len = len(prompt_tokens)
        labels[: prompt_len] = [-100] * prompt_len
        
        return {
            "input_ids": prompt_response_tokens,
            "labels": labels,
            "prompt_len": prompt_len,
        }

    return {"input_ids": prompt_response_tokens, "labels": labels}

# ======================================================
# 3. 可视化诊断函数
# ======================================================
def diagnose_masking(sample_data):
    print("\n" + "="*80)
    print(" 🧪 SFT 数据 Masking 真实对齐测试")
    print("="*80)

    # --- 调用核心函数 ---
    # 这完全模拟了 Dataset.map 里的行为
    processed = default_mdlm_sft_map_fn(
        sample_data, 
        tokenizer=tokenizer, 
        mask_prompt_loss=True
    )
    
    input_ids = processed['input_ids']
    labels = processed['labels']
    
    # --- 打印边界诊断 ---
    print(f"Input 总长度: {len(input_ids)}")
    print(f"Prompt 长度 : {processed['prompt_len']} (这些将被 Mask)")
    
    # 找到 Mask 和 Train 的交界处
    first_train_idx = processed['prompt_len']
    
    print("\n【交界处显微镜】(展示交界处前后的 Token)")
    print(f"{'位置':<6} | {'ID':<8} | {'Token (解码)':<25} | {'Label':<10} | {'状态'}")
    print("-" * 80)
    
    # 我们只看交界处前后 5 个 token，这最关键
    start_view = max(0, first_train_idx - 5)
    end_view = min(len(input_ids), first_train_idx + 10)
    
    for i in range(start_view, end_view):
        tid = input_ids[i]
        lbl = labels[i]
        
        token_str = tokenizer.decode([tid]).replace("\n", "\\n")
        # 缩略过长的字符串
        if len(token_str) > 20: token_str = token_str[:20] + "..."
        token_str = f"'{token_str}'"
        
        if lbl == -100:
            label_disp = "🚫 -100"
            status = "Masked (Prompt)"
        else:
            label_disp = f"✅ {lbl}"
            status = "Train (Answer)"
            
        # 高亮交界线
        if i == first_train_idx:
            print("-" * 80 + " <--- 训练开始线 (Loss Start)")
            
        print(f"{i:<6} | {tid:<8} | {token_str:<25} | {label_disp:<10} | {status}")
    
    print("-" * 80)

# ======================================================
# 4. 注入你的真实数据
# ======================================================
# 这是你之前给出的 TableDreamer 真实数据
real_data = {
    "item_id": "TableDreamer_train_data_1883", 
    "messages": [
        {
            "role": "user", 
            "content": "Convert all 'Initial Fuel' and 'Altered Fuel' data from liters to US gallons...\n(此处省略长文本)...\nOutput:"
        }, 
        {
            "role": "assistant", 
            "content": "To convert liters to US gallons, we use the conversion factor..."
        }
    ]
}

diagnose_masking(real_data)
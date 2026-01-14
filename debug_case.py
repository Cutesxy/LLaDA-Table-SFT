import torch
import os
import random
import numpy as np
from eval_core.models import LLaDAModelWrapper

# ================= 1. 配置区域 (严格对齐 Shell 脚本) =================

MODEL_PATH = "/home/zjusst/hxy/llada/models/GSAI-ML/LLaDA-8B-Instruct"
ADAPTER_PATH = None 

# [关键参数]
STEPS = 128         
GEN_LENGTH = 256    
SEED = 42           # <--- [新增] 严格对齐 main_eval.py 的默认值

# ================= 2. 调试数据 =================
debug_messages = [
    {
        "role": "user", 
        "content": """Table:
col: | year | song | album | position | chart |
| 1987 | "summergirls" | 24/7 | 50 | billboard hot 100 |
| 1989 | "24/7" | 24/7 | 42 | billboard hot 100 |
| 1989 | "24/7" | 24/7 | 12 | hot r&b/hip-hop songs |
| 1989 | "i like it" | 24/7 | 7 | billboard hot 100 |
| 1989 | "i like it" | 24/7 | 3 | hot dance club play |
| 1989 | "sunshine" | 24/7 | 23 | billboard hot 100 |
| 1990 | "never 2 much of u" | 24/7 | 61 | billboard hot 100 |
| 1990 | "romeo" | swingin' | 6 | billboard hot 100 |
| 1991 | "gentle" | swingin' | 31 | billboard hot 100 |
| 1993 | "ooh child" | the way i am | 27 | billboard hot 100 |
| 1993 | "endlessly" | the way i am | -- | billboard hot 100 |

Question: is the song summergirls on the album 24/7 or swingin'?

Let's think step by step. Analyze the table and the question, then provide the answer. At the end of your response, output the final result in this format: Answer: <result>"""
    }
]

def set_seed(seed):
    """
    [新增] 完全复刻 main_eval.py 的随机种子设置
    """
    print(f"Set Random Seed to: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_debug():
    # 1. 先设置种子，确保模型初始化的噪声一致
    set_seed(SEED)

    print("="*50)
    print(f"DEBUG CONFIGURATION (Aligned with run_eval.sh)")
    print(f"Model: {MODEL_PATH}")
    print(f"Steps: {STEPS}")
    print(f"Gen Length: {GEN_LENGTH}")
    print(f"Seed: {SEED}")
    print("="*50)

    try:
        model = LLaDAModelWrapper(
            model_path=MODEL_PATH,
            adapter_path=ADAPTER_PATH,
            steps=STEPS,
            gen_length=GEN_LENGTH
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n" + "-"*30)
    print("TEST 1: Running with SEED 42 + Shell Params...")
    print("-" * 30)
    
    # 这里的 max_new_tokens=GEN_LENGTH 会触发 models.py 里的逻辑
    output_1 = model.generate(debug_messages, max_new_tokens=GEN_LENGTH)
    
    print(f"\n[Raw Output]:\n{output_1}")
    print(f"\n[Output Length]: {len(output_1)} characters")
    
    if len(output_1.strip()) < 50: 
        print("\n[DIAGNOSIS]: Output is suspicious (Truncated).")
        print("Confirmed: Under SEED 42, the model fails to generate complete CoT.")
    else:
        print("\n[DIAGNOSIS]: Output looks normal in length.")
        print("Strange: Even with SEED 42, it works here but fails in main.")

if __name__ == "__main__":
    run_debug()
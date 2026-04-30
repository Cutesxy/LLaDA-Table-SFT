import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel 
import argparse

# ==========================================
# 1. 核心生成逻辑 (LLaDA Standard)
# ==========================================
def add_gumbel_noise(logits, temperature):
    if temperature == 0: return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    if gen_length < block_length: block_length = gen_length
    if gen_length % block_length != 0: block_length = gen_length 
    
    num_blocks = gen_length // block_length
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            mask_index = (x == mask_id)
            logits = model(x, attention_mask=attention_mask).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            # Remasking
            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    return x

# ==========================================
# 2. 交互式主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    # 请修改为你的实际模型路径
    parser.add_argument('--model_path', type=str, default="/home/zjusst/hxy/llada/models/GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument('--gen_length', type=int, default=512)
    parser.add_argument('--steps', type=int, default=256)
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 自动获取 mask_id
    mask_id = 126336
    if hasattr(model.config, 'mask_token_id') and model.config.mask_token_id is not None:
        mask_id = model.config.mask_token_id
    print(f"Using Mask ID: {mask_id}")

    print("\n" + "="*50)
    print(" LLADA-8B CHAT TEMPLATE TESTER ")
    print(" 输入 'exit' 或 'q' 退出")
    print("="*50 + "\n")

    # 对话历史 (可选：如果你想测试多轮对话，可以保留这个 list)
    # 这里为了简单测试，每次只测单轮
    
    while True:
        user_input = input("\n[User]: ")
        if user_input.lower() in ['exit', 'q']:
            break
        
        if not user_input.strip():
            continue

        # 构造消息
        messages = [{"role": "user", "content": user_input}]

        # >>>>>> 核心测试点：使用 apply_chat_template <<<<<<
        # add_generation_prompt=True 会让它自动补上 <|start_header_id|>assistant...
        prompt_str = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # [诊断打印] 即使跑崩了，也能看到崩在哪里
        print(f"\n[DEBUG] Template Output (Check the end!):\n{repr(prompt_str)}")

        # 转为 Tensor
        encoded = tokenizer(prompt_str, return_tensors='pt', add_special_tokens=False).to(device)
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        print(f"[DEBUG] Input Tokens: {input_ids.shape[1]}")
        print("Thinking...", end="", flush=True)

        # 生成
        out_tokens = generate(
            model=model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.gen_length,
            temperature=0.0,
            mask_id=mask_id
        )
        print(" Done!")

        # 解码
        generated_ids = out_tokens[:, input_ids.shape[1]:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print("\n" + "-"*20 + " Assistant " + "-"*20)
        print(response.strip())
        print("-" * 50)

if __name__ == "__main__":
    main()
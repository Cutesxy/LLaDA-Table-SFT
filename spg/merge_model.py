from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_path = '/home/william/yss/LLaDA-8B-Instruct'
peft_model_path = '/home/william/yss/SPG/spg/fsx-checkpoints/spg/table_fact_wtq_base_spg_mix_beta1.5_weight0.5_0202/checkpoint-2800'
merged_model_path = "/home/william/yss/spg-checkpoint/fact_wtq_sft-rl-checkpoint-2800"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path, trust_remote_code=True,torch_dtype="bfloat16", device_map="cuda")
peft_model = PeftModel.from_pretrained(model, peft_model_path, device_map="cuda")

merged_model = peft_model.merge_and_unload()
# 3. 保存合并后的模型权重
print(f"正在保存合并后的模型至: {merged_model_path}")
merged_model.save_pretrained(merged_model_path, safe_serialization=True)

# 4. 别忘了保存 tokenizer，这样以后加载时才完整
tokenizer.save_pretrained(merged_model_path)

print("保存完成！")

# 运行
# CUDA_VISIBLE_DEVICES=1 python merge_model.py
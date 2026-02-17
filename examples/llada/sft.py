import os
import sys
import json
import csv
import logging
import shutil
from dataclasses import dataclass, field
from functools import partial

import torch
import transformers
from transformers import TrainerCallback
from datasets import load_dataset, DatasetDict
from accelerate import PartialState

# 引入 dllm 库 (确保您的环境里已经安装或路径包含 dllm)
import dllm

# ============================================================
# 1. 全局日志设置
# ============================================================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("training.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

transformers.logging.set_verbosity_info()
logger = dllm.utils.get_default_logger(__name__)


# ============================================================
# 2. 核心修改：自定义结构化数据处理函数 (Flash Verifier Mode)
# ============================================================
def structural_flash_map_fn(example, tokenizer, max_length=1024):
    """
    [核心修改] 自定义的结构化数据映射函数
    功能：
    1. 提取 User 内容 (Prompt) 和 Assistant 内容 (Answer)。
    2. 不使用任何 Chat Template，直接进行由 BOS/EOS 包裹的拼接。
    3. 将 User 部分的 Label 设为 -100 (不计算 Loss)，只训练 Answer 部分。
    """
    messages = example["messages"]
    
    # 提取纯文本
    # messages[0] 是 User 输入 (包含 ... [Answer]\n)
    # messages[1] 是 Assistant 输出 (entailed / refuted)
    prompt_text = messages[0]["content"]
    answer_text = messages[1]["content"]

    # --- 手动 Tokenize (不加特殊 Token) ---
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    # --- 构造 Input IDs ---
    # 格式: [BOS] + [Prompt] + [Answer] + [EOS]
    input_ids = [tokenizer.bos_token_id] + prompt_ids + answer_ids + [tokenizer.eos_token_id]

    # --- 构造 Labels ---
    # Prompt 部分 (包括 BOS) 设为 -100 -> 告诉 Trainer 忽略这部分 Loss
    # Answer 部分 (包括 EOS) 保留 -> 只有这部分参与扩散去噪训练
    prompt_len = 1 + len(prompt_ids)  # 1 是 BOS 的长度
    labels = [-100] * prompt_len + answer_ids + [tokenizer.eos_token_id]

    # --- 截断处理 (防止超长) ---
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    
    # --- Attention Mask ---
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }


# ============================================================
# 3. 多卡安全的指标保存回调
# ============================================================
class SaveMetricsCallback(TrainerCallback):
    def __init__(self, output_dir: str):
        self.state = PartialState()
        self.is_main = self.state.is_local_main_process
        self.output_dir = output_dir

        if self.is_main:
            os.makedirs(output_dir, exist_ok=True)
            self.jsonl_path = os.path.join(output_dir, "trainer_metrics.jsonl")
            self.csv_path = os.path.join(output_dir, "trainer_metrics.csv")
        else:
            self.jsonl_path = None
            self.csv_path = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.is_main or logs is None:
            return
        
        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            **logs,
        }

        # Save JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Save CSV
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(record)


# ============================================================
# 4. 参数类定义
# ============================================================
@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = field(
        default="data/llada_sft_structural_flash.jsonl",
        metadata={"help": "Path to the training dataset file (jsonl)."},
    )
    eval_dataset_args: str = field(
        default=None,
        metadata={"help": "Path to the evaluation dataset file (jsonl)."},
    )
    load_preprocessed_data: bool = False
    # 注意：我们的 mask 逻辑已经在 structural_flash_map_fn 里硬编码了，
    # 所以这里的参数主要用于兼容接口，实际上已经被我们的自定义函数接管。
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )
    # [新增] 显式定义 max_length 以接收命令行参数
    max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length."},
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/LLaDA-Table-Flash-Verifier"
    group_by_length: bool = True
    evaluation_strategy: str = "steps"
    logging_strategy: str = "steps"
    logging_steps: int = 1
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 200
    logging_first_step: bool = True
    do_eval: bool = False  # 默认关，如果有 eval set 会自动开
    logging_dir: str = "logs"
    report_to: list[str] = field(default_factory=lambda: ["none"])


# ============================================================
# 5. 辅助诊断工具 (Debug)
# ============================================================
def debug_data_masking(dataset, tokenizer):
    """检查 Masking 是否符合预期：Prompt部分应为 -100"""
    logger.info("\n" + "="*40)
    logger.info(" [DEBUG] 正在检查数据 Masking (Structural Flash Mode)...")
    logger.info("="*40)
    
    try:
        sample = dataset[0]
        input_ids = sample['input_ids']
        labels = sample['labels']
        
        logger.info(f"Sample 0 Input Length: {len(input_ids)}")
        logger.info(f"Sample 0 Label Length: {len(labels)}")
        
        # 解码 Input
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
        logger.info(f"\n>>> [Full Input (First 200 chars)]:\n{decoded_input[:200]}...")
        
        # 解码 Label (仅看有效部分)
        active_labels = [l for l in labels if l != -100]
        decoded_labels = tokenizer.decode(active_labels, skip_special_tokens=False)
        
        logger.info(f"\n>>> [Trainable Answer (Expect 'entailed'/'refuted')]:\n[{decoded_labels}]")
        
        # 验证
        if len(active_labels) < 5: # 应该很短
            logger.info("\n[Pass] Masking 正常！模型只在训练 Answer 部分。")
        else:
            logger.warning("\n[Warning] Label 部分过长，请检查是否错误包含了 Prompt。")
            
    except Exception as e:
        logger.error(f"Debug check failed: {e}")
    logger.info("="*40 + "\n")


# ============================================================
# 6. 训练主流程
# ============================================================
def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_dataset_args is not None:
        training_args.do_eval = True

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)
    
    # 加载模型和 Tokenizer
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ---------------- Dataset 处理 ----------------
    with PartialState().local_main_process_first():
        logger.info("Loading datasets...")
        raw_datasets = DatasetDict()

        if data_args.dataset_args:
            raw_datasets["train"] = load_dataset(
                "json", data_files=data_args.dataset_args, split="train"
            )

        if data_args.eval_dataset_args:
            raw_datasets["test"] = load_dataset(
                "json", data_files=data_args.eval_dataset_args, split="train"
            )

        dataset = raw_datasets
        
        # [关键修改] 使用自定义的 structural_flash_map_fn
        if not data_args.load_preprocessed_data:
            logger.info(">>> 使用自定义 Structural Flash Map Function (Raw Text Concatenation)...")
            
            # [Fix] 获取 max_length 的稳健逻辑
            # 1. 优先从 data_args 获取 (命令行 --max_length 传入)
            # 2. 其次尝试 tokenizer 默认
            # 3. 兜底 4096
            _max_len = getattr(data_args, "max_length", None)
            if _max_len is None:
                _max_len = getattr(tokenizer, "model_max_length", 4096)
            if _max_len > 100000: # 某些 tokenizer 默认是 IntMax
                _max_len = 4096
                
            logger.info(f"Using max_length for mapping: {_max_len}")

            # 绑定 tokenizer 和 max_length
            map_fn = partial(
                structural_flash_map_fn, 
                tokenizer=tokenizer,
                max_length=_max_len, 
            )
            
            # 执行映射
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping dataset to Structural Flash format",
                remove_columns=["messages"], # 移除原始 messages 列，只保留 input_ids/labels
            )
        
        # 后处理（如果有的话，通常不需要）
        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    PartialState().wait_for_everyone()
    
    # 诊断数据
    if PartialState().is_local_main_process:
        debug_data_masking(dataset["train"], tokenizer)
    
    logger.info("Start training...")

    train_dataset = dataset["train"]
    eval_dataset = dataset.get("test", None)

    # ---------------- 初始化 Trainer ----------------
    # 注意：这里使用 dllm 库自带的 MDLMTrainer，它会自动处理 Masked Diffusion Loss
    # 只要我们的 label 里有 -100，它就会正确处理。
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        # 使用 NoAttentionMaskWrapper 配合 Seq2Seq Collator
        # 因为 LLaDA 是双向 Attention，通常 Attention Mask 全为 1
        data_collator=(
            dllm.utils.NoAttentionMaskWrapper(
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                    label_pad_token_id=-100, # 确保 pad 也被忽略
                )
            )
        ),
        callbacks=[
            SaveMetricsCallback(training_args.output_dir),
        ],
    )

    # 开始训练
    trainer.train()

    # ---------------- 保存模型 ----------------
    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    logger.info(f"Saving final model to {final_dir}...")
    trainer.save_model(final_dir)
    trainer.processing_class.save_pretrained(final_dir)

    # 复制自定义代码文件 (确保模型可移植)
    if PartialState().is_local_main_process:
        source_dir = model_args.model_name_or_path
        files_to_copy = ["configuration_llada.py", "modeling_llada.py"]
        for filename in files_to_copy:
            src = os.path.join(source_dir, filename)
            dst = os.path.join(final_dir, filename)
            if os.path.exists(src):
                shutil.copy(src, dst)

    logger.info("训练流程全部结束")

if __name__ == "__main__":
    train()
import os
import sys
import json
import csv
import logging
import shutil  #[新增] 用于复制文件
from dataclasses import dataclass, field
from functools import partial

import torch
import accelerate
from accelerate import PartialState
import transformers
from transformers import TrainerCallback
from datasets import load_dataset, DatasetDict
import dllm

# ============================================================
# 1. 全局日志（控制台 + training.log）
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
# 2. 多卡安全的 Trainer 指标保存 Callback
# ============================================================
class SaveMetricsCallback(TrainerCallback):
    """
    - 仅 rank0 写文件
    - 其他 rank 直接 return
    - 适配 accelerate / torchrun
    """

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
        if not self.is_main:
            return
        if logs is None:
            return

        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            **logs,
        }

        # ---------- JSONL ----------
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # ---------- CSV ----------
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(record)


# ============================================================
# 3. 参数定义
# ============================================================
@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = field(
        default="data/table_llada_train.jsonl",
        metadata={"help": "Path to the training dataset file (jsonl)."},
    )
    eval_dataset_args: str = field(
        default=None,
        metadata={"help": "Path to the evaluation dataset file (jsonl)."},
    )
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )


@dataclass
class TrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/LLaDA-Table-SFT-WithEval"
    group_by_length: bool = True

    # ===== 日志 / 评估 =====
    evaluation_strategy: str = "steps"
    logging_strategy: str = "steps"

    logging_steps: int = 1
    eval_steps: int = 5

    save_strategy: str = "steps"
    save_steps: int = 500
    logging_first_step: bool = True
    do_eval: bool = True

    logging_dir: str = "models/LLaDA-Table-SFT-WithEval/logs"

    # 不使用 tensorboard / wandb
    report_to: list[str] = field(default_factory=lambda: ["none"])

# ============================================================
# 4. 辅助诊断模块
# ============================================================
def debug_data_masking(dataset, tokenizer):
    """
    检查第一条数据的 Input 和 Label，确认 Mask 是否生效。
    """
    logger.info("\n" + "="*40)
    logger.info(" [DEBUG] 正在检查数据 Masking (Prompt Loss)...")
    logger.info("="*40)
    
    try:
        sample = dataset[0]
        input_ids = sample['input_ids']
        labels = sample['labels']
        
        # 1. 打印长度
        logger.info(f"Sample 0 Input Length: {len(input_ids)}")
        logger.info(f"Sample 0 Label Length: {len(labels)}")
        
        # 2. 解码 Input (模型看到的完整内容)
        decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
        logger.info(f"\n>>> [Full Input (First 300 chars)]:\n{decoded_input[:300]}...")
        
        # 3. 解码 Label (模型真正学习的内容)
        # 过滤掉 -100 的部分，只看有效 Label
        active_labels = [l for l in labels if l != -100]
        decoded_labels = tokenizer.decode(active_labels, skip_special_tokens=False)
        
        logger.info(f"\n>>> [Trainable Labels (First 300 chars)]:\n{decoded_labels[:300]}...")
        
        # 4. 自动诊断
        if len(active_labels) == 0:
            logger.error("\n[!!! 致命错误 !!!] Label 全是 -100！模型什么都学不到！请检查 DataMap 函数！")
        elif len(active_labels) == len(labels):
            logger.warning("\n[!!! 严重警告 !!!] Label 没有 Mask！模型在背诵 Prompt！loss_mask 参数可能失效！")
        else:
            logger.info("\n[Pass] Masking 看起来正常：Label 长度小于 Input，且只包含回答部分。")
            
    except Exception as e:
        logger.error(f"Debug check failed: {e}")
    logger.info("="*40 + "\n")

def check_lora_modules(model):
    """
    检查哪些层被 LoRA 激活了
    """
    logger.info("\n" + "="*40)
    logger.info(" [DEBUG] 正在检查 LoRA 目标模块...")
    logger.info("="*40)
    
    trainable_modules = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_modules.append(name)
    
    if not trainable_modules:
        logger.error("[!!! 致命错误 !!!] 模型没有任何可训练参数！LoRA 未生效！")
        return

    # 打印前 5 个和后 5 个可训练层
    logger.info(f"Total Trainable Parameters: {len(trainable_modules)}")
    logger.info(f"First 5 trainable modules: {trainable_modules[:5]}")
    
    # 简单的启发式检查
    has_mlp = any("up_proj" in n or "gate_proj" in n or "down_proj" in n for n in trainable_modules)
    if not has_mlp:
        logger.warning("\n[!!! 建议优化 !!!] LoRA 似乎只覆盖了 Attention 层 (q/v)。建议开启 target_modules=['all-linear'] 以提升表格推理能力！")
    else:
        logger.info("\n[Pass] LoRA 覆盖了 MLP 层，配置良好。")
    logger.info("="*40 + "\n")


# ============================================================
# 5. 训练主逻辑
# ============================================================
def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_dataset_args is not None:
        training_args.do_eval = True
        training_args.evaluation_strategy = "steps"

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)
    
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ---------------- Dataset ----------------
    with PartialState().local_main_process_first():
        logger.info("Loading datasets...")
        raw_datasets = DatasetDict()

        if data_args.dataset_args:
            logger.info(f"Loading train dataset from {data_args.dataset_args}")
            raw_datasets["train"] = load_dataset(
                "json", data_files=data_args.dataset_args, split="train"
            )

        if data_args.eval_dataset_args:
            logger.info(f"Loading eval dataset from {data_args.eval_dataset_args}")
            raw_datasets["test"] = load_dataset(
                "json", data_files=data_args.eval_dataset_args, split="train"
            )

        dataset = raw_datasets
        
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                dllm.utils.default_mdlm_sft_map_fn,
                tokenizer=tokenizer,
                mask_prompt_loss=data_args.mask_prompt_loss,
            )
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc,
                desc="Mapping dataset to SFT format",
            )

        dataset = dllm.utils.post_process_dataset(dataset, data_args)

    PartialState().wait_for_everyone()
    
    # 只有主进程执行检查
    if PartialState().is_local_main_process:
        debug_data_masking(dataset["train"], tokenizer)
        # 如果是全量微调，这里可能打印出来是全部参数，如果是LoRA则只有部分
        check_lora_modules(model) 
    
    logger.info("Start training...")

    train_dataset = dataset["train"]
    eval_dataset = dataset.get("test", dataset.get("validation", None))

    # ---------------- Trainer ----------------
    trainer = dllm.core.trainers.MDLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=(
            dllm.utils.NoAttentionMaskWrapper(
                transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding=True,
                    label_pad_token_id=tokenizer.pad_token_id,
                )
            )
        ),
        callbacks=[
            SaveMetricsCallback(training_args.output_dir),
        ],
    )

    trainer.train()

    # ---------------- Save final ----------------
    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    trainer.processing_class.save_pretrained(final_dir)

    # [新增] 仅在全量微调时，复制自定义代码文件
    # ============================================================
    if PartialState().is_local_main_process:
        # 如果使用了 PEFT (LoRA)，通常不需要复制 Python 代码
        # 但既然你特别提到是 SFT 全量微调，我们这里强制进行复制检查
        logger.info(f"正在尝试从源目录复制自定义代码文件到: {final_dir} ...")
        
        source_dir = model_args.model_name_or_path
        files_to_copy = ["configuration_llada.py", "modeling_llada.py"]
        
        copied_count = 0
        for filename in files_to_copy:
            src_file = os.path.join(source_dir, filename)
            dst_file = os.path.join(final_dir, filename)
            
            try:
                if os.path.exists(src_file):
                    shutil.copy(src_file, dst_file)
                    logger.info(f"  [Success] 已复制: {filename}")
                    copied_count += 1
                else:
                    logger.warning(f"  [Warning] 源文件中没找到: {src_file}。如果是标准 HF 模型，这很正常；如果是自定义模型，请检查。")
            except Exception as e:
                logger.error(f"  [Error] 复制 {filename} 失败: {e}")
        
        if copied_count == 2:
            logger.info("成功复制了所有必要的 LLaDA 代码文件。")

    logger.info("训练流程全部结束！")

# ============================================================
# 6. Entry
# ============================================================
if __name__ == "__main__":
    train()
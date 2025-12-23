import os
import sys
import json
import csv
import logging
from dataclasses import dataclass, field
from functools import partial

import accelerate
import transformers
from transformers import TrainerCallback
from datasets import load_dataset, DatasetDict
import dllm


# ============================================================
# 1. 日志系统：控制台 + 文件（training.log）
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
# 2. Trainer 日志回调：保存 loss / eval_loss / lr
# ============================================================
class SaveMetricsCallback(TrainerCallback):
    """
    将 Trainer 的训练指标保存为：
    - JSONL（推荐，结构化、便于分析）
    - CSV（Excel 直接可读）
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.jsonl_path = os.path.join(output_dir, "trainer_metrics.jsonl")
        self.csv_path = os.path.join(output_dir, "trainer_metrics.csv")

    def on_log(self, args, state, control, logs=None, **kwargs):
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
    output_dir: str = "models/LLaDA-Table-SFT"
    group_by_length: bool = True

    # ===== 日志 & 评估 =====
    evaluation_strategy: str = "steps"
    logging_strategy: str = "steps"

    logging_steps: int = 1
    eval_steps: int = 5

    save_strategy: str = "steps"
    save_steps: int = 500
    logging_first_step: bool = True
    do_eval: bool = True

    # HF Trainer 自己的 logging 目录（即使不用 tensorboard 也建议保留）
    logging_dir: str = "models/LLaDA-Table-SFT/logs"

    # 不启用 tensorboard / wandb
    report_to: list[str] = field(default_factory=lambda: ["none"])


# ============================================================
# 4. 训练主函数
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
    with accelerate.PartialState().local_main_process_first():
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

    accelerate.PartialState().wait_for_everyone()
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


# ============================================================
# 5. Entry
# ============================================================
if __name__ == "__main__":
    train()

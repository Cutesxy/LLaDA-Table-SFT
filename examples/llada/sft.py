import os
import sys
from dataclasses import dataclass, field
from functools import partial
import logging  # [新增 1] 导入 logging

import accelerate
import transformers
from datasets import load_dataset, DatasetDict
import dllm

# [新增 2] 配置日志：同时输出到 控制台(Stream) 和 文件(training.log)
# 放在最开头执行，这样能捕获后续所有库的日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("training.log", mode='w'), # 写入文件
        logging.StreamHandler(sys.stdout)              # 输出到屏幕
    ]
)

# 保持 INFO 级别，确保能看到基础信息
transformers.logging.set_verbosity_info()
logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Base"


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = field(
        default="data/table_llada_train.jsonl",
        metadata={"help": "Path to the training dataset file (jsonl)."}
    )
    eval_dataset_args: str = field(
        default=None,
        metadata={"help": "Path to the evaluation dataset file (jsonl)."}
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
    
    # === 关键配置 ===
    evaluation_strategy: str = "steps"
    logging_strategy: str = "steps"
    
    logging_steps: int = 1         # 每 1 步打印
    eval_steps: int = 5            # 每 5 步评估
    
    save_strategy: str = "steps"
    save_steps: int = 500
    logging_first_step: bool = True
    do_eval: bool = True
    
    # 改回 none，不需要 tensorboard，直接输出到屏幕 (现在也会被上面的 logging 配置捕获进文件)
    report_to: list[str] = field(default_factory=lambda: ["none"])


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
    
    with accelerate.PartialState().local_main_process_first():
        logger.info("Loading datasets...")
        raw_datasets = DatasetDict()

        if data_args.dataset_args:
            logger.info(f"Loading train dataset from {data_args.dataset_args}...")
            raw_datasets["train"] = load_dataset("json", data_files=data_args.dataset_args, split="train")

        if data_args.eval_dataset_args:
            logger.info(f"Loading eval dataset from {data_args.eval_dataset_args}...")
            raw_datasets["test"] = load_dataset("json", data_files=data_args.eval_dataset_args, split="train")

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
    eval_dataset = dataset.get("test", None)
    if eval_dataset is None:
        eval_dataset = dataset.get("validation", None)

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
                ),
            )
        ),
    )
    
    trainer.train()
    
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
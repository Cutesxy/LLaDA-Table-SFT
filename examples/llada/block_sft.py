import os
import sys
import json
import csv
import logging
import shutil
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import accelerate
from accelerate import PartialState
import transformers
from transformers import TrainerCallback
from datasets import load_dataset, DatasetDict

# 引入 dllm 基础库
import dllm
import dllm.utils
import dllm.core.trainers 
# 注意：这里我们不再 import dllm.bdlm，而是直接在下方定义相关类

# ============================================================
# 0. BDLM 相关类定义 (直接集成到脚本中)
# ============================================================

# 尝试导入 CollatorWrapper，如果 dllm 版本不同可能位置不一样
try:
    from dllm.utils.collators import CollatorWrapper
except ImportError:
    # 如果找不到，提供一个简单的透传基类
    class CollatorWrapper:
        def __init__(self, collator, tokenizer):
            self.collator = collator
            self.tokenizer = tokenizer
        def __call__(self, features):
            features = self.before(features)
            batch = self.collator(features)
            return self.after(batch)
        def before(self, features): return features
        def after(self, batch): return batch

# ============================================================
# [修复] 适配 dllm 库的 CollatorWrapper 接口
# ============================================================
class AppendEOSBlockWrapper(CollatorWrapper):
    def __init__(self, collator, tokenizer, block_size: int = 32):
        # 1. 既然报错说父类只收 2 个参数 (self, collator)，那我们就只传 collator
        try:
            super().__init__(collator)
        except TypeError:
            #以此兼容万一父类需要 tokenizer 的情况 (虽然根据报错应该不需要)
            super().__init__(collator, tokenizer)

        # 2. 我们自己手动保存 tokenizer，确保后面能用到
        self.tokenizer = tokenizer 
        self.block_size = block_size

    def before(self, features):
        for ex in features:
            ids = ex["input_ids"]
            labs = ex["labels"]

            assert isinstance(ids, list) and isinstance(labs, list)

            L = len(ids)
            # 计算对齐目标长度
            target = (L + self.block_size - 1) // self.block_size * self.block_size
            pad_len = target - L
            if pad_len > 0:
                # 补充 EOS
                # 注意：这里使用了 self.tokenizer，所以上面必须确保它被保存了
                ex["input_ids"] = ids + [self.tokenizer.eos_token_id] * pad_len
                ex["labels"] = labs + [self.tokenizer.eos_token_id] * pad_len
        return features

def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask.
    """
    # Indicate whether token belongs to xt or x0
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    # Compute block indices
    block_q = torch.where(
        x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size
    )
    block_kv = torch.where(
        x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size
    )

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal

# 继承自 dllm.core.trainers.MDLMTrainer
class BD3LMTrainer(dllm.core.trainers.MDLMTrainer):
    def __init__(
        self,
        block_size: int = 32,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.block_size = block_size

def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: list[dict[str, Any]],
        return_outputs: bool = False,
        **kwargs,
    ):
        # 兼容性处理
        if hasattr(self.processing_class, "padding_side"):
            assert self.processing_class.padding_side == "right"
        
        inputs = self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape

        # === 1. Sample diffusion timesteps ===
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        alpha_t = self.scheduler(t)
        p_mask = 1.0 - alpha_t.unsqueeze(1).expand(b, l)

        # === 2. Apply stochastic masking ===
        masked_indices = (torch.rand((b, l), device=input_ids.device) < p_mask) & (
            labels != -100
        )
        noised_input_ids = torch.where(
            masked_indices, self.processing_class.mask_token_id, input_ids
        )

        # === 3. Forward pass (block-diffusion) ===
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)

        # 获取模型 config
        unwrapped_model = self.accelerator.unwrap_model(model)
        attn_impl = getattr(unwrapped_model.config, "_attn_implementation", "sdpa")

        if attn_impl == "sdpa" or attn_impl == "eager":
            attention_mask = block_diff_mask(
                b=None,
                h=None,
                q_idx=torch.arange(l * 2)[:, None],
                kv_idx=torch.arange(l * 2)[None, :],
                block_size=self.block_size,
                n=l,
            )
            attention_mask = (
                attention_mask.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l, 2 * l)
            )
            attention_mask = attention_mask.to(input_ids.device)
            
        elif attn_impl == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask
            attention_mask = create_block_mask(
                partial(block_diff_mask, block_size=self.block_size, n=l),
                B=None, H=None, Q_LEN=l * 2, KV_LEN=l * 2,
            )
        else:
            raise NotImplementedError(f"Attention implementation {attn_impl} not supported yet.")

        # [修改] 既然模型报错不支持 position_ids，我们这里不再手动构建和传入它
        # base_pos = torch.arange(l, device=input_ids.device).unsqueeze(0).expand(b, l)
        # concat_position_ids = torch.cat([base_pos, base_pos], dim=1)

        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            # position_ids=concat_position_ids,  # <--- [FIX] 注释掉这一行以解决报错
        )
        
        if hasattr(self, "_postprocess_outputs"):
            outputs = self._postprocess_outputs(outputs)
        
        logits = outputs.logits
        logits = logits[:, :l]

        # === 4. Handle degenerate cases ===
        if not masked_indices.any():
            zero_loss = logits.sum() * 0.0
            return (zero_loss, outputs) if return_outputs else zero_loss

        # === 5. Compute loss weights ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_indices=masked_indices
        )

        # === 6. Compute weighted cross-entropy ===
        token_loss = F.cross_entropy(
            logits[masked_indices], input_ids[masked_indices], reduction="none"
        )
        token_loss = token_loss * loss_weights[masked_indices]

        # === 7. Normalize ===
        effective_lengths = torch.sum(labels != -100, dim=1, keepdim=True).expand(b, l)
        loss = torch.sum(token_loss / effective_lengths[masked_indices]) / b

        return (loss, outputs) if return_outputs else loss

# ============================================================
# 1. 全局日志
# ============================================================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("training_bdlm.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

transformers.logging.set_verbosity_info()
logger = dllm.utils.get_default_logger(__name__)


# ============================================================
# 2. 回调函数
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

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.is_main or logs is None:
            return
        
        record = {
            "step": state.global_step,
            "epoch": state.epoch,
            **logs,
        }

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

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
class BDLMTrainingArguments(dllm.utils.TrainingArguments):
    output_dir: str = "models/LLaDA-Table-SFT-BDLM"
    group_by_length: bool = True
    block_size: int = field(
        default=32,
        metadata={"help": "Block size for Block Diffusion."},
    )
    evaluation_strategy: str = "steps"
    logging_strategy: str = "steps"
    logging_steps: int = 1
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 500
    logging_first_step: bool = True
    do_eval: bool = True
    logging_dir: str = "models/LLaDA-Table-SFT-BDLM/logs"
    report_to: list[str] = field(default_factory=lambda: ["none"])


# ============================================================
# 4. 辅助诊断
# ============================================================
def debug_bdlm_data(dataset, block_size):
    logger.info("\n" + "="*40)
    logger.info(f" [DEBUG] BDLM 数据检查 (Block Size: {block_size})")
    logger.info("="*40)
    try:
        sample = dataset[0]
        input_ids = sample['input_ids']
        logger.info(f"Sample 0 Raw Length: {len(input_ids)}")
        
        target = (len(input_ids) + block_size - 1) // block_size * block_size
        pad_len = target - len(input_ids)
        
        logger.info(f"Expected Padding: {pad_len}")
        logger.info(f"Target Length: {target}")
    except Exception as e:
        logger.error(f"Debug check failed: {e}")
    logger.info("="*40 + "\n")


# ============================================================
# 5. 训练主逻辑
# ============================================================
def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, BDLMTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_dataset_args is not None:
        training_args.do_eval = True

    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)
    
    model = dllm.utils.get_model(model_args=model_args)
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # Dataset
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
    
    if PartialState().is_local_main_process:
        debug_bdlm_data(dataset["train"], training_args.block_size)
    
    logger.info("Start training...")

    train_dataset = dataset["train"]
    eval_dataset = dataset.get("test", dataset.get("validation", None))

    # Collator Pipeline
    base_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id, 
    )
    no_attn_collator = dllm.utils.NoAttentionMaskWrapper(base_collator)
    bdlm_collator = AppendEOSBlockWrapper(
        no_attn_collator, 
        tokenizer=tokenizer, 
        block_size=training_args.block_size
    )

    trainer = BD3LMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=bdlm_collator,
        callbacks=[SaveMetricsCallback(training_args.output_dir)],
        block_size=training_args.block_size,
    )

    trainer.train()

    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    trainer.processing_class.save_pretrained(final_dir)

    # 复制自定义文件
    if PartialState().is_local_main_process:
        source_dir = model_args.model_name_or_path
        files_to_copy = ["configuration_llada.py", "modeling_llada.py"]
        for filename in files_to_copy:
            src_file = os.path.join(source_dir, filename)
            dst_file = os.path.join(final_dir, filename)
            if os.path.exists(src_file):
                try:
                    shutil.copy(src_file, dst_file)
                except:
                    pass

    logger.info("Done.")

if __name__ == "__main__":
    train()
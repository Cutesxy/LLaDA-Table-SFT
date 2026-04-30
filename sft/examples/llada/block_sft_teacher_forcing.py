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
                # 对齐补出的 token 只服务于 block 切分，不参与监督
                ex["labels"] = labs + [-100] * pad_len
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
class BD3LMTeacherForcingTrainer(dllm.core.trainers.MDLMTrainer):
    def __init__(
        self,
        block_size: int = 32,
        tf_max_stages: int = 4,
        tf_confidence_threshold: float = 0.7,
        tf_only_masked_tokens: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if tf_max_stages < 1:
            raise ValueError("tf_max_stages must be >= 1")
        self.block_size = block_size
        self.tf_max_stages = tf_max_stages
        self.tf_confidence_threshold = tf_confidence_threshold
        self.tf_only_masked_tokens = tf_only_masked_tokens

    @staticmethod
    def _zero_loss_with_grad(model):
        # Build a scalar zero that stays connected to autograd graph.
        p = next(model.parameters(), None)
        if p is None:
            return torch.tensor(0.0, requires_grad=True)
        return p.float().sum() * 0.0

    def _build_bd_attention(self, model, seq_len, device):
        unwrapped_model = self.accelerator.unwrap_model(model)
        attn_impl = getattr(unwrapped_model.config, "_attn_implementation", "sdpa")
        attention_mask = None
        attention_bias = None

        if attn_impl == "sdpa" or attn_impl == "eager":
            block_mask = block_diff_mask(
                b=None,
                h=None,
                q_idx=torch.arange(seq_len * 2, device=device)[:, None],
                kv_idx=torch.arange(seq_len * 2, device=device)[None, :],
                block_size=self.block_size,
                n=seq_len,
            )
            # LLaDA eager path expects structural mask via attention_bias.
            attention_bias = block_mask.unsqueeze(0).unsqueeze(0)
        elif attn_impl == "flex_attention":
            from torch.nn.attention.flex_attention import create_block_mask

            attention_mask = create_block_mask(
                partial(block_diff_mask, block_size=self.block_size, n=seq_len),
                B=None,
                H=None,
                Q_LEN=seq_len * 2,
                KV_LEN=seq_len * 2,
            )
        else:
            raise NotImplementedError(
                f"Attention implementation {attn_impl} not supported yet."
            )

        return attention_mask, attention_bias

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

        # === 3. Build block attention mask once per batch ===
        attention_mask, attention_bias = self._build_bd_attention(
            model=model,
            seq_len=l,
            device=input_ids.device,
        )

        # === 4. Decide which tokens participate in TF scheduling ===
        if self.tf_only_masked_tokens:
            valid_loss_mask = masked_indices
        else:
            valid_loss_mask = labels != -100

        # Keep execution path consistent across ranks:
        # if masked set is empty, fall back to all supervised tokens.
        if not valid_loss_mask.any():
            valid_loss_mask = labels != -100

        # === 5. Compute loss weights ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_indices=masked_indices
        )
        outputs = None

        # Keep track of tokens already selected by previous stages.
        already_selected_mask = ~valid_loss_mask

        # === 6. Multi-stage confidence teacher forcing ===
        for stage in range(self.tf_max_stages):
            is_last_stage = stage == (self.tf_max_stages - 1)
            current_input_ids = noised_input_ids.clone()

            # Teacher forcing: replace selected x_t positions with GT tokens.
            gt_replacement_mask = already_selected_mask & valid_loss_mask
            current_input_ids[gt_replacement_mask] = input_ids[gt_replacement_mask]

            concat_input_ids = torch.cat([current_input_ids, input_ids], dim=1)
            with torch.set_grad_enabled(is_last_stage):
                outputs = model(
                    input_ids=concat_input_ids,
                    attention_mask=attention_mask,
                    attention_bias=attention_bias,
                )

            if hasattr(self, "_postprocess_outputs"):
                outputs = self._postprocess_outputs(outputs)

            logits = outputs.logits[:, :l]
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_labels = labels.reshape(-1)

            valid_and_unselected = (valid_loss_mask & ~already_selected_mask).reshape(-1)

            # P(GT): gather ground-truth token confidence at each position.
            safe_flat_labels = flat_labels.clone()
            safe_flat_labels[safe_flat_labels < 0] = 0
            with torch.no_grad():
                if valid_and_unselected.any():
                    probs = F.softmax(flat_logits.detach(), dim=-1)
                    gt_probs = probs.gather(
                        dim=1, index=safe_flat_labels.unsqueeze(-1)
                    ).squeeze(-1)
                    confidence_selection_mask_flat = (
                        gt_probs >= self.tf_confidence_threshold
                    ) & valid_and_unselected
                    confidence_selection_mask = confidence_selection_mask_flat.view_as(labels)
                else:
                    gt_probs = torch.zeros_like(flat_logits[:, 0])
                    confidence_selection_mask = torch.zeros_like(labels, dtype=torch.bool)

            stage_selection_mask = confidence_selection_mask.clone()
            num_blocks = (l + self.block_size - 1) // self.block_size

            for block_idx in range(num_blocks):
                start_idx = block_idx * self.block_size
                end_idx = min((block_idx + 1) * self.block_size, l)
                block_slice = slice(start_idx, end_idx)

                block_valid_tokens = valid_loss_mask[:, block_slice].sum().item()
                if block_valid_tokens == 0:
                    continue

                # Ceil schedule guarantees full coverage by the last stage.
                required_cumulative = (
                    block_valid_tokens * (stage + 1) + self.tf_max_stages - 1
                ) // self.tf_max_stages

                already_selected_in_block = (
                    (already_selected_mask[:, block_slice] & valid_loss_mask[:, block_slice])
                    .sum()
                    .item()
                )
                confidence_selected_in_block = (
                    (confidence_selection_mask[:, block_slice] & valid_loss_mask[:, block_slice])
                    .sum()
                    .item()
                )

                needed = (
                    required_cumulative
                    - already_selected_in_block
                    - confidence_selected_in_block
                )
                if needed <= 0:
                    continue

                padding_candidates_block = (
                    valid_loss_mask[:, block_slice]
                    & ~already_selected_mask[:, block_slice]
                    & ~confidence_selection_mask[:, block_slice]
                )
                cand_batch, cand_block_idx = torch.where(padding_candidates_block)
                if cand_batch.numel() == 0:
                    continue

                cand_global_flat_idx = cand_batch * l + (start_idx + cand_block_idx)
                cand_probs = gt_probs[cand_global_flat_idx]
                sorted_local = torch.argsort(cand_probs, descending=True)

                num_to_select = min(needed, sorted_local.numel())
                if num_to_select <= 0:
                    continue

                selected_global_flat_idx = cand_global_flat_idx[sorted_local[:num_to_select]]
                selected_batch = selected_global_flat_idx // l
                selected_seq = selected_global_flat_idx % l
                stage_selection_mask[selected_batch, selected_seq] = True

            # Only include new valid tokens this stage.
            stage_selection_mask = (
                stage_selection_mask & valid_loss_mask & ~already_selected_mask
            )
            if not is_last_stage:
                if stage_selection_mask.any():
                    already_selected_mask = already_selected_mask | stage_selection_mask
                continue

            # Last stage: only keep one autograd graph to avoid OOM.
            # If the current stage selects nothing, fall back to all remaining valid tokens.
            if not stage_selection_mask.any():
                stage_selection_mask = valid_loss_mask & ~already_selected_mask
            if not stage_selection_mask.any():
                stage_selection_mask = valid_loss_mask

            token_loss_all = F.cross_entropy(
                flat_logits,
                safe_flat_labels,
                reduction="none",
            ).view_as(labels)
            weighted_token_loss = token_loss_all * loss_weights
            loss = weighted_token_loss[stage_selection_mask].mean()
            if not loss.requires_grad:
                loss = self._zero_loss_with_grad(model)
            return (loss, outputs) if return_outputs else loss

        # Fallback: if we exited before reaching the gradient-enabled last stage,
        # run one final grad pass so loss always has a valid grad_fn.
        current_input_ids = noised_input_ids.clone()
        gt_replacement_mask = already_selected_mask & valid_loss_mask
        current_input_ids[gt_replacement_mask] = input_ids[gt_replacement_mask]
        concat_input_ids = torch.cat([current_input_ids, input_ids], dim=1)
        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
        )
        if hasattr(self, "_postprocess_outputs"):
            outputs = self._postprocess_outputs(outputs)

        logits = outputs.logits[:, :l]
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        safe_flat_labels = flat_labels.clone()
        safe_flat_labels[safe_flat_labels < 0] = 0

        stage_selection_mask = valid_loss_mask & ~already_selected_mask
        if not stage_selection_mask.any():
            stage_selection_mask = valid_loss_mask

        token_loss_all = F.cross_entropy(
            flat_logits,
            safe_flat_labels,
            reduction="none",
        ).view_as(labels)
        weighted_token_loss = token_loss_all * loss_weights
        loss = weighted_token_loss[stage_selection_mask].mean()
        if not loss.requires_grad:
            loss = self._zero_loss_with_grad(model)

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
    tf_max_stages: int = field(
        default=4,
        metadata={"help": "Number of staged teacher-forcing refinement passes."},
    )
    tf_confidence_threshold: float = field(
        default=0.7,
        metadata={"help": "Confidence threshold P(GT) for selecting tokens in a stage."},
    )
    tf_only_masked_tokens: bool = field(
        default=True,
        metadata={"help": "If true, TF scheduling is only applied on diffusion-masked tokens."},
    )


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
        label_pad_token_id=-100,
    )
    no_attn_collator = dllm.utils.NoAttentionMaskWrapper(base_collator)
    bdlm_collator = AppendEOSBlockWrapper(
        no_attn_collator, 
        tokenizer=tokenizer, 
        block_size=training_args.block_size
    )

    trainer = BD3LMTeacherForcingTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=bdlm_collator,
        callbacks=[SaveMetricsCallback(training_args.output_dir)],
        block_size=training_args.block_size,
        tf_max_stages=training_args.tf_max_stages,
        tf_confidence_threshold=training_args.tf_confidence_threshold,
        tf_only_masked_tokens=training_args.tf_only_masked_tokens,
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

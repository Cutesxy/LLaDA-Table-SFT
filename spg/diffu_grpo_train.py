# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import wandb
import os
import importlib.util
from dataclasses import MISSING
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from trl.trainer.grpo_config import GRPOConfig as TRLGRPOConfig
from peft import LoraConfig
try:
    from swanlab.integration.transformers import SwanLabCallback
except Exception:
    SwanLabCallback = None


# Custom imports
from diffu_grpo_trainer import DiffuGRPOTrainer
from spg_trainer import SPGTrainer
from diffu_grpo_config import DiffuGRPOConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
    reward_len,
    tablegpt_reward_func,
    table_fact_wtq_reward_func,
)
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    get_sudoku_questions_new,
    set_random_seed,
    get_math_questions,
    get_tablegpt_questions,
    get_table_fact_wtq_questions
)

def _backfill_grpo_config_compat(args):
    """
    Backfill missing fields when local DiffuGRPOConfig lags behind TRL's GRPOConfig.
    This avoids repeated AttributeError after TRL upgrades.
    """
    for name, field_def in TRLGRPOConfig.__dataclass_fields__.items():
        if hasattr(args, name):
            continue
        if field_def.default is not MISSING:
            setattr(args, name, field_def.default)
        elif field_def.default_factory is not MISSING:  # type: ignore[attr-defined]
            setattr(args, name, field_def.default_factory())  # type: ignore[misc]
        else:
            setattr(args, name, None)

    # Practical compatibility fallbacks.
    if not hasattr(args, "max_tool_calling_iterations"):
        setattr(args, "max_tool_calling_iterations", None)
    if not hasattr(args, "num_generations_eval"):
        setattr(args, "num_generations_eval", None)
    if getattr(args, "num_generations_eval", None) is None:
        setattr(args, "num_generations_eval", getattr(args, "num_generations", None))

    # Keep consistent with newer TRL constraints.
    if getattr(args, "sync_ref_model", False) and float(getattr(args, "beta", 0.0)) == 0.0:
        setattr(args, "sync_ref_model", False)

    return args


def _normalize_runtime_grpo_args(args):
    """Normalize critical GRPO runtime args to avoid sampler/runtime crashes."""
    # Ensure positive integers.
    for name, default in (
        ("per_device_train_batch_size", 1),
        ("num_generations", 1),
        ("steps_per_generation", 1),
    ):
        v = getattr(args, name, default)
        if v is None or int(v) <= 0:
            setattr(args, name, default)

    # TRL 1.1 sampler can hit batch_size=0 when num_generations > per_device_train_batch_size.
    pbs = int(getattr(args, "per_device_train_batch_size", 1))
    ng = int(getattr(args, "num_generations", 1))
    if ng > pbs:
        print(
            f"[WARN] num_generations ({ng}) > per_device_train_batch_size ({pbs}); "
            f"auto-adjust num_generations -> {pbs} to avoid sampler ZeroDivisionError."
        )
        setattr(args, "num_generations", pbs)
        if getattr(args, "num_generations_eval", None) is None or int(getattr(args, "num_generations_eval")) > pbs:
            setattr(args, "num_generations_eval", pbs)

    gbs = getattr(args, "generation_batch_size", None)
    if gbs is not None:
        gbs = int(gbs)
        if gbs <= 0:
            setattr(args, "generation_batch_size", None)
        else:
            ng_now = int(getattr(args, "num_generations", 1))
            # TRL sampler may compute per-step mini-batch as generation_batch_size // num_generations.
            # If this is 0, it triggers ZeroDivisionError.
            if gbs < ng_now:
                print(
                    f"[WARN] generation_batch_size ({gbs}) < num_generations ({ng_now}); "
                    f"auto-adjust generation_batch_size -> {ng_now}."
                )
                setattr(args, "generation_batch_size", ng_now)

    return args

def main(grpo_config, model_config):

    # Set seed for reproducibility
    set_random_seed(grpo_config.seed)

    # Load dataset based on configuration
    if grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif grpo_config.dataset == "tablegpt":
        dataset = get_tablegpt_questions()
        reward_functions = [tablegpt_reward_func]
    elif grpo_config.dataset == "table_fact_wtq":
        dataset = get_table_fact_wtq_questions()
        reward_functions = [table_fact_wtq_reward_func]
    elif grpo_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    # elif grpo_config.dataset == "sudoku":
    #     dataset = get_sudoku_questions()
    #     reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "sudoku_new":
        dataset = get_sudoku_questions_new(few_shot=grpo_config.few_shot)
        reward_functions = [sudoku_reward_func]
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]

    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=grpo_config.seed)

    # Split dataset if needed
    if grpo_config.dataset in ["countdown", "sudoku", "sudoku_new"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        grpo_config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            grpo_config.model_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(grpo_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )

    callbacks = []
    swan_mode = str(os.getenv("SWANLAB_MODE", "")).lower()
    swan_enabled = swan_mode not in {"", "off", "none", "disabled", "false", "0"}
    if swan_mode == "local" and importlib.util.find_spec("swanboard") is None:
        print("[WARN] SWANLAB_MODE=local but `swanboard` is not installed; disabling SwanLab callback.")
        swan_enabled = False
    if SwanLabCallback is not None and swan_enabled:
        try:
            callbacks.append(SwanLabCallback())
        except Exception as e:
            print(f"[WARN] SwanLab disabled due to init error: {e}")
    else:
        print("[INFO] SwanLab callback disabled.")

    if grpo_config.trainer == "diffu_grpo":
        # Initialize and run trainer
        trainer = DiffuGRPOTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            callbacks=callbacks,
        )
    elif grpo_config.trainer == "spg":
        trainer = SPGTrainer(
            args=grpo_config,
            model=model,
            peft_config=peft_config,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            callbacks=callbacks,
        )
    else:
        raise ValueError(f"Invalid trainer: {grpo_config.trainer}")

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((DiffuGRPOConfig, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config()
    grpo_config = _backfill_grpo_config_compat(grpo_config)
    grpo_config = _normalize_runtime_grpo_args(grpo_config)
    grpo_config.report_to="none"
    main(grpo_config=grpo_config, model_config=model_config)

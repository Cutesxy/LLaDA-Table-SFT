"""
References:

Simple and Effective Masked Diffusion Language Models:
https://arxiv.org/abs/2406.07524

Large Language Diffusion Models:
https://arxiv.org/abs/2502.09992
"""

from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler
from dllm.utils.data import prepend_bos


class MDLMTrainer(transformers.Trainer):

    def __init__(
        self,
        scheduler: BaseAlphaScheduler | None = None,
        time_epsilon: float = 1e-3,
        loss_weight_type: str = "scheduler",  # "ones"
        right_shift_logits: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not (0.0 < time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")

        self.scheduler = scheduler or LinearAlphaScheduler()
        self.time_epsilon = time_epsilon
        self.loss_weight_type = loss_weight_type
        self.right_shift_logits = right_shift_logits
        
        # [新增] 初始化时清空一下旧的 debug 日志文件
        if self.is_world_process_zero():
            with open("eval_debug.log", "w") as f:
                f.write("=== Start Evaluation Log ===\n")

    def _preprocess_inputs(self, inputs):
        if self.right_shift_logits:
            labels = inputs.get("labels", None)
            if labels is not None:
                if torch.all(labels[:, 0] == -100):
                    return inputs

            inputs = prepend_bos(
                inputs,
                bos_token_id=self.processing_class.bos_token_id,
                label_pad_token_id=-100,
            )
        return inputs

    def _postprocess_outputs(self, outputs):
        if self.right_shift_logits:
            logits = outputs.logits
            outputs.logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return outputs

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        b, l = inputs["input_ids"].shape
        if self.loss_weight_type == "scheduler":
            loss_weights = self.scheduler.weight(t).unsqueeze(1).repeat(1, l)
        elif self.loss_weight_type == "ones":
            loss_weights = torch.ones_like(inputs["input_ids"])
        else:
            raise NotImplementedError
        return loss_weights

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()

        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()

        return (loss.detach(), logits, labels)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
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
        p_mask = 1 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # === 2. Apply stochastic masking ===
        masked_indices = (torch.rand((b, l), device=input_ids.device) < p_mask) & (
            labels != -100
        )
        noised_input_ids = torch.where(
            masked_indices, self.processing_class.mask_token_id, input_ids
        )

        # === 3. Forward pass ===
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        outputs = self._postprocess_outputs(outputs)
        logits = outputs.logits

        # === 4. Handle degenerate cases ===
        if not masked_indices.any():
            return (
                (logits.sum() * 0.0, outputs) if return_outputs else logits.sum() * 0.0
            )

        # === 5. Compute weights ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_indices=masked_indices
        )

        # === 6. Compute loss ===
        assert (input_ids[masked_indices] == labels[masked_indices]).all()
        token_loss = F.cross_entropy(
            logits[masked_indices], input_ids[masked_indices], reduction="none"
        )
        token_loss = token_loss * loss_weights[masked_indices]

        # === 7. Normalize ===
        effective_lengths = torch.sum(labels != -100, dim=1, keepdim=True).expand(b, l)
        loss = torch.sum(token_loss / effective_lengths[masked_indices]) / b

        # =========================================================================
        # [修改] 强行写入文件 eval_debug.log
        # =========================================================================
        # if self.is_world_process_zero() and not model.training:
        #     # 使用追加模式 'a' 打开文件，写完立即关闭，最安全
        #     with open("eval_debug.log", "a") as f:
        #         f.write(f"[Eval] Batch Loss: {loss.item():.4f}\n")
        # # =========================================================================

        # === 8. Return ===
        return (loss, outputs) if return_outputs else loss
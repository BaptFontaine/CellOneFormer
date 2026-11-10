"""
Model wrapper for OneFormer instance segmentation on microscopy images.
Supports fine-tuning, query adjustment, and inference post-processing.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from transformers import (
    OneFormerConfig,
    OneFormerForUniversalSegmentation,
    OneFormerProcessor,
)


class CellOneFormerModule(nn.Module):
    """
    Wrapper around Hugging Face's OneFormerForUniversalSegmentation tailored for cell instances.
    """

    def __init__(
        self,
        pretrained_model_name: str = "shi-labs/oneformer_ade20k_swin_tiny",
        num_classes: int = 2,
        num_queries: int = 150,
        freeze_backbone: bool = False,
        is_training: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.is_training = is_training

        print(f"[Model] Loading OneFormer config and weights from '{pretrained_model_name}' (is_training={is_training})...")
        
        # Load processor
        self.processor = OneFormerProcessor.from_pretrained(pretrained_model_name)
        if hasattr(self.processor, "image_processor"):
            ip = self.processor.image_processor
            ip.ignore_index = 0
            ip.do_reduce_labels = False
            # Hugging Face OneFormer requires num_text to be an integer (queries count)
            ip.num_text = num_queries
            # Explicitly register class 1 (cell) as a countable "thing" instance
            ip.thing_ids = [1]
            if hasattr(ip, "class_names") and isinstance(ip.class_names, list):
                if "cell" not in ip.class_names:
                    ip.class_names.append("cell")
            if hasattr(ip, "metadata") and isinstance(ip.metadata, dict):
                ip.metadata["thing_ids"] = [1]
                ip.metadata["1"] = "cell"
                ip.metadata[1] = "cell"
                ip.metadata["0"] = "background"
                ip.metadata[0] = "background"
                if "class_names" in ip.metadata and isinstance(ip.metadata["class_names"], list):
                    if "cell" not in ip.metadata["class_names"]:
                        ip.metadata["class_names"].append("cell")

        # Load configuration and customize for cell segmentation (2 classes: background, cell)
        config = OneFormerConfig.from_pretrained(
            pretrained_model_name,
            is_training=is_training,
        )
        config.num_queries = num_queries
        config.num_labels = num_classes
        config.is_training = is_training
        
        # Label dictionary
        config.id2label = {0: "background", 1: "cell"}
        config.label2id = {"background": 0, "cell": 1}

        # Load model with customized head
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            pretrained_model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        # Bypass auxiliary text-contrastive loss (not used for 2-class instance segmentation)
        try:
            object.__setattr__(self.model.config, "contrastive_temperature", None)
        except Exception:
            pass
        if hasattr(self.model, "criterion"):
            self.model.criterion.loss_contrastive = lambda *args, **kwargs: {}

        if freeze_backbone:
            print("[Model] Freezing backbone layers for transfer learning...")
            for param in self.model.model.pixel_level_module.encoder.parameters():
                param.requires_grad = False

        # Precompute default tokenized task_inputs for instance segmentation
        task_seq_len = getattr(self.processor, "task_seq_len", 77)
        task_tokens = self.processor.tokenizer(
            ["the task is instance"],
            padding="max_length",
            max_length=task_seq_len,
            return_tensors="pt",
        )["input_ids"]
        self.register_buffer("default_task_inputs", task_tokens)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.Tensor] = None,
        text_inputs: Optional[torch.Tensor] = None,
        task_inputs: Optional[torch.Tensor] = None,
        mask_labels: Optional[List[torch.Tensor]] = None,
        class_labels: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass during training or inference.
        Returns loss during training if mask_labels and class_labels are provided.
        """
        # Ensure task_inputs is never None (OneFormer requires a tokenized task prompt)
        if task_inputs is None:
            task_inputs = self.default_task_inputs.repeat(pixel_values.shape[0], 1)
        elif task_inputs.ndim == 1:
            task_inputs = task_inputs.unsqueeze(0)
            if task_inputs.shape[0] == 1 and pixel_values.shape[0] > 1:
                task_inputs = task_inputs.repeat(pixel_values.shape[0], 1)

        # Prevent OneFormerModel from requiring text_inputs during eval or when text_inputs is omitted
        prev_cfg_train = getattr(self.model.config, "is_training", False)
        prev_mdl_train = getattr(getattr(self.model, "model", None), "is_training", False)
        active_text_training = bool(self.training and text_inputs is not None and mask_labels is not None)
        self.model.config.is_training = active_text_training
        if hasattr(self.model, "model"):
            self.model.model.is_training = active_text_training

        try:
            outputs = self.model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                text_inputs=text_inputs if active_text_training else None,
                task_inputs=task_inputs,
                mask_labels=mask_labels,
                class_labels=class_labels,
                **kwargs,
            )
        finally:
            self.model.config.is_training = prev_cfg_train
            if hasattr(self.model, "model"):
                self.model.model.is_training = prev_mdl_train

        return outputs

    def post_process_instances(
        self,
        outputs: Any,
        target_sizes: List[tuple],
        threshold: float = 0.2,
        mask_threshold: float = 0.5,
        min_area: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Direct transformer query decoding into binary instance masks.
        Bypasses ADE20K pre-trained metadata dictionaries for robust custom 2-class cell segmentation.
        """
        batch_size = outputs.class_queries_logits.shape[0]
        results = []

        for b in range(batch_size):
            target_h, target_w = target_sizes[b]
            class_logits = outputs.class_queries_logits[b]  # (num_queries, num_classes + 1)
            mask_logits = outputs.masks_queries_logits[b:b+1]  # (1, num_queries, H_m, W_m)

            probs = torch.softmax(class_logits, dim=-1)
            # Class 1 corresponds to "cell"
            cell_scores = probs[:, 1]

            # Upsample predicted mask logits to target crop dimensions
            upsampled_masks = torch.nn.functional.interpolate(
                mask_logits,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )[0].sigmoid()  # (num_queries, target_h, target_w)

            masks_list = []
            scores_list = []

            # Sort candidate queries by descending confidence
            sorted_indices = torch.argsort(cell_scores, descending=True)
            for idx in sorted_indices:
                score = float(cell_scores[idx].item())
                if score < threshold:
                    break
                bin_mask = (upsampled_masks[idx] >= mask_threshold).detach().cpu().numpy()
                area = int(bin_mask.sum())
                if area >= min_area:
                    masks_list.append(bin_mask)
                    scores_list.append(score)

            results.append({
                "masks": masks_list,
                "scores": scores_list,
            })

        return results

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
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries

        print(f"[Model] Loading OneFormer config and weights from '{pretrained_model_name}'...")
        
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
            is_training=True,
        )
        config.num_queries = num_queries
        config.num_labels = num_classes
        config.is_training = True
        
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
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            text_inputs=text_inputs,
            task_inputs=task_inputs,
            mask_labels=mask_labels,
            class_labels=class_labels,
            **kwargs,
        )
        return outputs

    def post_process_instances(
        self,
        outputs: Any,
        target_sizes: List[tuple],
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Post-processes raw transformer query outputs into binary instance masks.
        """
        return self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )

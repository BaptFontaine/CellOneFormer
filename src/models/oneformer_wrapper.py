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
            self.processor.image_processor.ignore_index = 0
            self.processor.image_processor.do_reduce_labels = False

        # Load configuration and customize for cell segmentation (2 classes: background, cell)
        config = OneFormerConfig.from_pretrained(pretrained_model_name)
        config.num_queries = num_queries
        config.num_labels = num_classes
        
        # Label dictionary
        config.id2label = {0: "background", 1: "cell"}
        config.label2id = {"background": 0, "cell": 1}

        # Load model with customized head
        self.model = OneFormerForUniversalSegmentation.from_pretrained(
            pretrained_model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

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

"""
PyTorch Dataset and Data Collator for OneFormer Multi-Modality Cell Segmentation.
Compatible with Hugging Face OneFormerProcessor.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Optional, Any
import albumentations as A

from src.dataset.tiler import load_microscopy_image, load_instance_mask


def get_microscopy_augmentations(is_train: bool = True, image_size: int = 512) -> A.Compose:
    """
    Returns data augmentations tailored for multi-modality cell microscopy.
    Cells are rotationally invariant.
    """
    if is_train:
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.06, 0.06), rotate=(-45, 45), p=0.5),
            A.GaussNoise(p=0.3),
        ])
    return A.Compose([])


class CellOneFormerDataset(Dataset):
    """
    Dataset wrapping tiled microscopy crops for OneFormer training and validation.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: Optional[str] = None,
        processor: Optional[Any] = None,
        task: str = "instance",
        is_train: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.processor = processor
        self.task = task
        self.is_train = is_train
        self.transforms = get_microscopy_augmentations(is_train=is_train)

        # Collect image files
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        self.image_paths.sort()

        if max_samples is not None and max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {images_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        image = load_microscopy_image(img_path)

        mask = None
        if self.masks_dir is not None:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Match mask pattern: name_mask.png, name.png, etc.
            possible_names = [
                f"{base_name}_mask.png",
                f"{base_name}.png",
                f"{base_name}_mask.tif",
                f"{base_name}.tif",
            ]
            for p_name in possible_names:
                m_path = os.path.join(self.masks_dir, p_name)
                if os.path.exists(m_path):
                    mask = load_instance_mask(m_path)
                    break

        # Apply spatial & radiometric augmentations
        if mask is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Ensure image is PIL Image with 3 channels
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        pil_image = Image.fromarray(image.astype(np.uint8))

        # Hugging Face OneFormerProcessor strictly expects "instance", "semantic", or "panoptic"
        task_name = self.task.lower().strip()
        if "instance" in task_name:
            task_key = "instance"
        elif "semantic" in task_name:
            task_key = "semantic"
        else:
            task_key = "panoptic"

        # If processor is available, preprocess for OneFormer
        if self.processor is not None:
            if mask is not None:
                # Build instance_id_to_semantic_id dictionary:
                # 0 is background (class 0), and all non-zero instance IDs are cells (class 1)
                unique_ids = np.unique(mask)
                instance_id_to_semantic_id = {int(uid): 1 for uid in unique_ids}
                instance_id_to_semantic_id[0] = 0

                inputs = self.processor(
                    images=pil_image,
                    task_inputs=[task_key],
                    segmentation_maps=mask,
                    instance_id_to_semantic_id=instance_id_to_semantic_id,
                    return_tensors="pt",
                )
                # Squeeze leading batch dimension added by processor
                item = {k: v.squeeze(0) for k, v in inputs.items()}
                item["image_path"] = img_path
                return item
            else:
                inputs = self.processor(
                    images=pil_image,
                    task_inputs=[task_key],
                    return_tensors="pt",
                )
                item = {k: v.squeeze(0) for k, v in inputs.items()}
                item["image_path"] = img_path
                return item

        return {
            "image": torch.tensor(image).permute(2, 0, 1).float() / 255.0,
            "mask": torch.tensor(mask).long() if mask is not None else None,
            "task": task_key,
            "image_path": img_path,
        }


def collate_fn_oneformer(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collator for OneFormer batches handling variable-length mask labels.
    """
    keys = batch[0].keys()
    collated = {}

    for k in keys:
        if k in ("image_path", "task"):
            collated[k] = [b[k] for b in batch]
        elif k in ("mask_labels", "class_labels"):
            # Variable number of instances per image
            collated[k] = [b[k] for b in batch]
        else:
            collated[k] = torch.stack([b[k] for b in batch])

    return collated

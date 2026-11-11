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
import cv2

from src.dataset.tiler import load_microscopy_image, load_instance_mask


def get_microscopy_augmentations(is_train: bool = True, image_size: int = 512) -> A.Compose:
    """
    Returns data augmentations tailored for multi-modality cell microscopy.
    Cells are rotationally invariant; photometric transforms improve cross-modality generalization
    (brightfield, fluorescence, phase-contrast, DIC).
    """
    if is_train:
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.06, 0.06),
                rotate=(-45, 45),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.05, p=0.3),
            A.GaussNoise(p=0.25),
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
        all_image_paths = []
        for ext in extensions:
            all_image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        all_image_paths.sort()

        # Filter for images that have matching masks during training
        if self.masks_dir is not None and self.is_train:
            valid_paths = []
            for img_p in all_image_paths:
                b_name = os.path.splitext(os.path.basename(img_p))[0]
                for p_name in [f"{b_name}_mask.png", f"{b_name}.png", f"{b_name}_mask.tif", f"{b_name}.tif"]:
                    mp = os.path.join(self.masks_dir, p_name)
                    if os.path.exists(mp):
                        valid_paths.append(img_p)
                        break
            if len(valid_paths) > 0:
                self.image_paths = valid_paths
        else:
            self.image_paths = all_image_paths

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

        # Apply spatial & radiometric augmentations (cast mask to uint16 for OpenCV warpAffine safety)
        if mask is not None:
            orig_image, orig_mask = image.copy(), mask.copy()
            augmented = self.transforms(image=image, mask=mask.astype(np.uint16))
            image = augmented["image"]
            mask = augmented["mask"].astype(np.int32)
            # If geometric transform pushed edge cells out of frame, restore un-transformed pair
            if mask.max() == 0 and orig_mask.max() > 0:
                image, mask = orig_image, orig_mask
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

        # If processor is available, preprocess image and build exact instance mask_labels/class_labels directly
        if self.processor is not None:
            inputs = self.processor(
                images=pil_image,
                task_inputs=[task_key],
                return_tensors="pt",
            )
            item = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    item[k] = v.squeeze(0)
                elif isinstance(v, list) and len(v) > 0:
                    item[k] = v[0]
                else:
                    item[k] = v

            if mask is not None:
                _, h_pv, w_pv = item["pixel_values"].shape
                if mask.shape[0] != h_pv or mask.shape[1] != w_pv:
                    mask_resized = cv2.resize(
                        mask.astype(np.uint16),
                        (w_pv, h_pv),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.int32)
                else:
                    mask_resized = mask

                instance_ids = [int(uid) for uid in np.unique(mask_resized) if uid > 0]
                if len(instance_ids) == 0 and self.is_train:
                    # Fallback single small instance only if tile is completely empty
                    fallback = np.zeros((h_pv, w_pv), dtype=np.float32)
                    cv2.circle(fallback, (w_pv // 2, h_pv // 2), 12, 1.0, -1)
                    item["mask_labels"] = torch.from_numpy(fallback).unsqueeze(0)
                    item["class_labels"] = torch.ones(1, dtype=torch.long)
                elif len(instance_ids) > 0:
                    bin_masks = np.stack(
                        [(mask_resized == uid).astype(np.float32) for uid in instance_ids],
                        axis=0,
                    )
                    item["mask_labels"] = torch.from_numpy(bin_masks)
                    item["class_labels"] = torch.ones(len(instance_ids), dtype=torch.long)

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
    # Filter out any corrupt items that lack mask_labels during training
    valid_batch = [b for b in batch if "mask_labels" in b and "pixel_values" in b]
    if len(valid_batch) == 0:
        valid_batch = batch

    keys = valid_batch[0].keys()
    collated = {}

    for k in keys:
        if k in ("image_path", "task"):
            collated[k] = [b.get(k) for b in valid_batch]
        elif k in ("mask_labels", "class_labels"):
            collated[k] = [b[k] for b in valid_batch if k in b]
        else:
            tensors = [b[k] for b in valid_batch if k in b and isinstance(b[k], torch.Tensor)]
            if len(tensors) == len(valid_batch):
                collated[k] = torch.stack(tensors)
            else:
                collated[k] = [b.get(k) for b in valid_batch]

    return collated

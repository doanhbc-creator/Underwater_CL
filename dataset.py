from __future__ import annotations
import albumentations as A

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset

BASE_DIR = Path(__file__).resolve().parent / "Underwater_CL"
ALL_IMG_PATH = BASE_DIR / "data" / "all_imgs"
ALL_MASK_PATH = BASE_DIR / "data" / "all_masks"
TASK_CONFIG_PATH = BASE_DIR / "data" / "task_configs"

# training set images augmentation
def get_training_augmentation(seed: Optional[int] = None):
    train_transform = [
        A.Resize(height=512, width=512),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform, seed=seed)


def get_validation_augmentation(seed: Optional[int] = None):
    """Resize inputs to a fixed shape for stable batching and evaluation."""
    test_transform = [
        A.Resize(height=512, width=512),
    ]
    return A.Compose(test_transform, seed=seed)

class Dataset(BaseDataset):
    def __init__(
        self,
        metadata,
        label_mapping,
        class_values,
        augmentation=None,
    ):
        self.metadata = metadata
        self.augmentation = augmentation
        self.label_mapping = label_mapping
        self.class_values = class_values

    def __getitem__(self, i):
        image = cv2.imread(self.metadata[i]['image'])
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.metadata[i]['image']}")

        # BGR-->RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.metadata[i]['mask'], 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {self.metadata[i]['mask']}")

        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        ignore_index = 255
        mapped_mask = np.full_like(mask, ignore_index, dtype=np.uint8)

        for old_value, new_value in self.label_mapping.items():
            mapped_mask[mask == int(old_value)] = new_value
        
        mask = mapped_mask

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image.transpose(2, 0, 1), mask

    def __len__(self):
        return len(self.metadata)

def _resolve_path(path: Optional[Union[str, Path]], default_dir: Path) -> Path:
    if path is None:
        return default_dir

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return default_dir / candidate


def get_task_config_paths(task_config_dir: Optional[Union[str, Path]] = None) -> List[Path]:
    config_dir = Path(task_config_dir) if task_config_dir is not None else TASK_CONFIG_PATH
    return sorted(config_dir.glob("*.json"))


TASK_NAMES = [path.stem for path in get_task_config_paths()]


def load_task_configs(
    task_config_dir: Optional[Union[str, Path]] = None,
    task_name: Optional[Union[str, Iterable[str]]] = None,
) -> List[Dict[str, Any]]:
    config_paths = get_task_config_paths(task_config_dir)
    selected_names = {task_name} if isinstance(task_name, str) else set(task_name or [])

    configs: List[Dict[str, Any]] = []
    for config_path in config_paths:
        config_name = config_path.stem
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        config_title = config.get("task_name")
        config_id = str(config.get("task_id"))
        if selected_names and config_name not in selected_names and config_title not in selected_names and config_id not in selected_names:
            continue


        config["_config_path"] = str(config_path)
        configs.append(config)

    return configs


def load_task_samples(
    task_name: Optional[Union[str, Iterable[str]]] = None,
    split: str = "train",
    image_dir: Optional[Union[str, Path]] = None,
    mask_dir: Optional[Union[str, Path]] = None,
    task_config_dir: Optional[Union[str, Path]] = None,
    require_existing_files: bool = True,
) -> List[Dict[str, Any]]:
    image_root = _resolve_path(image_dir, ALL_IMG_PATH)
    mask_root = _resolve_path(mask_dir, ALL_MASK_PATH)

    samples: List[Dict[str, Any]] = []
    for config in load_task_configs(task_config_dir=task_config_dir, task_name=task_name):
        task_config_name = config.get("task_name") or Path(config["_config_path"]).stem
        split_data = config.get("data", {}).get(split, [])

        for sample in split_data:
            image_path = _resolve_path(sample.get("image"), image_root)
            mask_path = _resolve_path(sample.get("mask"), mask_root)

            if require_existing_files and not image_path.exists():
                continue
            if require_existing_files and not mask_path.exists():
                continue

            samples.append(
                {
                    "task_name": task_config_name,
                    "task_id": config.get("task_id"),
                    "source": sample.get("source"),
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "label_mapping": config.get("label_mapping"),
                    "datasets_used": config.get("datasets_used"),
                }
            )

    return samples


def load_task_datasets(
    task_name: Optional[Union[str, Iterable[str]]] = None,
    split: str = "train",
    image_dir: Optional[Union[str, Path]] = None,
    mask_dir: Optional[Union[str, Path]] = None,
    task_config_dir: Optional[Union[str, Path]] = None,
    require_existing_files: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    datasets: Dict[str, List[Dict[str, Any]]] = {}

    for config in load_task_configs(task_config_dir=task_config_dir, task_name=task_name):
        task_config_name = config.get("task_name") or Path(config["_config_path"]).stem
        split_data = config.get("data", {}).get(split, [])
        task_samples: List[Dict[str, Any]] = []

        for sample in split_data:
            image_path = _resolve_path(sample.get("image"), _resolve_path(image_dir, ALL_IMG_PATH))
            mask_path = _resolve_path(sample.get("mask"), _resolve_path(mask_dir, ALL_MASK_PATH))

            if require_existing_files and not image_path.exists():
                continue
            if require_existing_files and not mask_path.exists():
                continue

            task_samples.append(
                {
                    "task_name": task_config_name,
                    "task_id": config.get("task_id"),
                    "source": sample.get("source"),
                    "image": str(image_path),
                    "mask": str(mask_path),
                    "label_mapping": config.get("label_mapping"),
                    "datasets_used": config.get("datasets_used"),
                }
            )

        datasets[task_config_name] = task_samples

    return datasets

__all__ = [
    "ALL_IMG_PATH",
    "ALL_MASK_PATH",
    "TASK_CONFIG_PATH",
    "TASK_NAMES",
    "get_task_config_paths",
    "load_task_configs",
    "load_task_samples",
    "load_task_datasets",
]
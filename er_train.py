from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from dataset import (
    Dataset,
    get_training_augmentation,
    get_validation_augmentation,
    load_task_datasets,
)
from models.er_model import SegmentationModel

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints_er"
CHECKPOINT_DIR.mkdir(exist_ok=True)

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


set_seed(SEED)


def _seed_worker(worker_id: int):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _build_label_mapping(metadata):
    task_label_mapping = metadata[0]["label_mapping"]
    mapping_name = list(task_label_mapping.keys())[0]
    label_mapping = task_label_mapping[mapping_name]
    return {
        int(old_value): int(new_value)
        for old_value, new_value in label_mapping.items()
        if old_value != "_comment"
    }

def _load_previous_weights(model, checkpoint_path):
    if not checkpoint_path.exists():
        return 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    filtered_state = {}
    model_state = model.state_dict()

    def _load_partial_output_channels(key, src_tensor, dst_tensor):
        if src_tensor.ndim == 0 or dst_tensor.ndim == 0:
            return None
        if src_tensor.shape[1:] != dst_tensor.shape[1:]:
            return None

        num_loaded = min(src_tensor.shape[0], dst_tensor.shape[0])
        if num_loaded == 0:
            return None

        loaded_tensor = dst_tensor.clone()
        loaded_tensor[:num_loaded] = src_tensor[:num_loaded]
        return loaded_tensor

    classification_head_keys = {
        "model.segmentation_head.0.weight",
        "model.segmentation_head.0.bias",
    }

    for key, value in checkpoint.items():
        if key not in model_state:
            continue

        if model_state[key].shape == value.shape:
            filtered_state[key] = value
            continue

        if key in classification_head_keys:
            partial = _load_partial_output_channels(key, value, model_state[key])
            if partial is not None:
                filtered_state[key] = partial

    if filtered_state:
        model.load_state_dict(filtered_state, strict=True)
    return len(filtered_state)


def _evaluate_iou(model, dataset, num_classes, device):
    model = model.to(device)
    model.eval()
    loader = dataset if isinstance(dataset, DataLoader) else DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    predictions = []
    targets = []
    with torch.no_grad():
        for image, mask in loader:
            image = image.to(device, dtype=torch.float32)
            mask = mask.long().to(device)
            logits = model(image)
            pred_mask = logits.softmax(dim=1).argmax(dim=1)
            predictions.append(pred_mask.cpu())
            targets.append(mask.cpu())

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    tp, fp, fn, tn = smp.metrics.get_stats(
        predictions,
        targets,
        mode="multiclass",
        num_classes=num_classes,
    )
    return smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()


def main():
    train_datasets = load_task_datasets(split="train", require_existing_files=True)
    val_datasets = load_task_datasets(split="val", require_existing_files=True)
    test_datasets = load_task_datasets(split="test", require_existing_files=True)

    task_names = list(train_datasets.keys())
    all_class_values = []
    previous_checkpoint = None
    task_iou_history = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    er_buffer_state = None
    for task_idx, task in enumerate(task_names):
        print(f"Training on task: {task}")
        set_seed(SEED + task_idx)
        label_mapping = _build_label_mapping(train_datasets[task])
        task_class_values = sorted({int(value) for value in label_mapping.values()})
        cumulative_class_values = sorted(set(all_class_values) | set(task_class_values))
        all_class_values = cumulative_class_values

        train_dataset = Dataset(
            metadata=train_datasets[task],
            label_mapping=label_mapping,
            class_values=cumulative_class_values,
            augmentation=get_training_augmentation(seed=SEED + task_idx),
        )

        valid_dataset = Dataset(
            metadata=val_datasets[task],
            label_mapping=label_mapping,
            class_values=cumulative_class_values,
            augmentation=get_validation_augmentation(seed=SEED + task_idx),
        )

        test_dataset = Dataset(
            metadata=test_datasets[task],
            label_mapping=label_mapping,
            class_values=cumulative_class_values,
            augmentation=get_validation_augmentation(seed=SEED + task_idx),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(SEED + task_idx),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(SEED + task_idx),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(SEED + task_idx),
        )

        model = SegmentationModel(
            arch="Segformer",
            encoder_name="resnet34",
            in_channels=3,
            out_classes=len(cumulative_class_values),
            er_enabled=True,
            er_buffer_size=128,
            er_minibatch_size=8,
        ).to(device)

        if previous_checkpoint is not None:
            loaded_count = _load_previous_weights(model, previous_checkpoint)
            print(f"Loaded {loaded_count} matching weights from {previous_checkpoint.name}")

        if er_buffer_state is not None:
            model.load_er_buffer_state(er_buffer_state)

        EPOCHS = 50
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            log_every_n_steps=1,
            enable_checkpointing=False
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
        er_buffer_state = model.get_er_buffer_state()

        checkpoint_path = CHECKPOINT_DIR / f"{task}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "er_buffer_state": model.get_er_buffer_state(),
            },
            checkpoint_path,
        )
        previous_checkpoint = checkpoint_path
        print(f"Saved checkpoint to {checkpoint_path}")

        current_task_iou = _evaluate_iou(model, test_loader, len(cumulative_class_values), device)
        print(f"{task} test IoU: {current_task_iou:.4f}")

        task_iou_history[task] = {}
        for eval_task in task_names[: task_names.index(task) + 1]:
            eval_label_mapping = _build_label_mapping(train_datasets[eval_task])
            eval_dataset = Dataset(
                metadata=test_datasets[eval_task],
                label_mapping=eval_label_mapping,
                class_values=cumulative_class_values,
                augmentation=get_validation_augmentation(seed=SEED + task_idx),
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=8,
                shuffle=False,
                num_workers=0,
                worker_init_fn=_seed_worker,
                generator=torch.Generator().manual_seed(SEED + task_idx),
            )
            iou = _evaluate_iou(model, eval_loader, len(cumulative_class_values), device)
            task_iou_history[task][eval_task] = iou

        print("IoU across seen tasks:")
        for eval_task, iou in task_iou_history[task].items():
            print(f"  - {eval_task}: {iou:.4f}")

    history_path = CHECKPOINT_DIR / "task_iou_history_er.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(task_iou_history, handle, indent=2)
    print(f"Saved IoU history to {history_path}")

if __name__ == "__main__":
    main()
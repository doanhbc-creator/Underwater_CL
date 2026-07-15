from __future__ import annotations

import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    Dataset,
    get_validation_augmentation,
    load_task_datasets,
)
from models.derpp_model import SegmentationModel


CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
VISUALIZATION_DIR = CHECKPOINT_DIR / "visualizations_by_image"
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_SAMPLES_PER_TASK = 10   # change this to visualize more images per task


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


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name))


def _build_label_mapping(metadata):
    task_label_mapping = metadata[0]["label_mapping"]
    mapping_name = list(task_label_mapping.keys())[0]
    label_mapping = task_label_mapping[mapping_name]

    return {
        int(old_value): int(new_value)
        for old_value, new_value in label_mapping.items()
        if old_value != "_comment"
    }


def _get_task_class_values(metadata):
    label_mapping = _build_label_mapping(metadata)
    return sorted({int(value) for value in label_mapping.values()})


def _build_cumulative_class_values(train_datasets, task_names):
    cumulative_class_values_by_task = []
    all_class_values = []

    for task in task_names:
        task_class_values = _get_task_class_values(train_datasets[task])
        all_class_values = sorted(set(all_class_values) | set(task_class_values))
        cumulative_class_values_by_task.append(list(all_class_values))

    global_class_values = sorted(
        {
            class_value
            for class_values in cumulative_class_values_by_task
            for class_value in class_values
        }
    )

    return cumulative_class_values_by_task, global_class_values


def _extract_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

    return checkpoint


def _load_previous_weights(model, checkpoint_path: Path):
    if not checkpoint_path.exists():
        print(f"[Warning] Checkpoint not found: {checkpoint_path}")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint = _extract_checkpoint_state_dict(checkpoint)

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Invalid checkpoint format: {checkpoint_path}")

    filtered_state = {}
    model_state = model.state_dict()

    def _load_partial_output_channels(src_tensor, dst_tensor):
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

    segmentation_head_keys = {
        "model.segmentation_head.0.weight",
        "model.segmentation_head.0.bias",
    }

    for key, value in checkpoint.items():
        if key not in model_state:
            continue

        if model_state[key].shape == value.shape:
            filtered_state[key] = value
            continue

        if key in segmentation_head_keys:
            partial = _load_partial_output_channels(value, model_state[key])
            if partial is not None:
                filtered_state[key] = partial

    if filtered_state:
        model_state.update(filtered_state)
        model.load_state_dict(model_state, strict=True)

    return len(filtered_state)


def _build_model(num_classes: int, device):
    model = SegmentationModel(
        arch="Segformer",
        encoder_name="resnet34",
        in_channels=3,
        out_classes=num_classes,
        derpp_enabled=True,
        derpp_buffer_size=128,
        derpp_minibatch_size=8,
        derpp_alpha=0.5,
        derpp_beta=1.0,
    )

    return model.to(device)


def _ensure_image_tensor(image):
    if torch.is_tensor(image):
        return image

    image = torch.from_numpy(image)

    if image.ndim == 3 and image.shape[-1] in [1, 3]:
        image = image.permute(2, 0, 1)

    return image


def _ensure_mask_tensor(mask):
    if torch.is_tensor(mask):
        return mask.long()

    return torch.from_numpy(mask).long()


def _tensor_to_image(image_tensor):
    image = image_tensor.detach().cpu()

    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = image.permute(1, 2, 0)

    image = image.numpy().astype(np.float32)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    image_min = image.min()
    image_max = image.max()

    if image_max > 1.0 or image_min < 0.0:
        image = (image - image_min) / (image_max - image_min + 1e-8)

    return image


def _local_index_mask_to_display_mask(
    index_mask,
    local_class_values,
    global_class_values,
):
    """
    Convert local class index mask into global display index mask.

    Example:
    local_class_values = [0, 2, 5]
    global_class_values = [0, 1, 2, 3, 4, 5]

    local index 0 -> class value 0 -> display index 0
    local index 1 -> class value 2 -> display index 2
    local index 2 -> class value 5 -> display index 5
    """
    if torch.is_tensor(index_mask):
        index_mask = index_mask.detach().cpu().numpy()

    display_mask = np.zeros_like(index_mask, dtype=np.int64)

    global_value_to_display_idx = {
        int(class_value): display_idx
        for display_idx, class_value in enumerate(global_class_values)
    }

    for local_idx, class_value in enumerate(local_class_values):
        class_value = int(class_value)
        display_idx = global_value_to_display_idx[class_value]
        display_mask[index_mask == local_idx] = display_idx

    return display_mask


def _overlay_mask_on_image(image, display_mask, num_display_classes, alpha=0.45):
    image = image.copy()
    display_mask = np.asarray(display_mask)

    cmap = plt.get_cmap("tab20", max(num_display_classes, 2))
    colored_mask = cmap(display_mask)[..., :3]

    foreground = display_mask > 0

    overlay = image.copy()
    overlay[foreground] = (
        (1.0 - alpha) * image[foreground]
        + alpha * colored_mask[foreground]
    )

    return overlay


def _predict_single_image(model, image_tensor, device):
    model.eval()

    image_tensor = _ensure_image_tensor(image_tensor)
    image_tensor = image_tensor.unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        logits = model(image_tensor)
        pred_mask = logits.softmax(dim=1).argmax(dim=1)

    return pred_mask[0].detach().cpu()


def _save_trajectory_figure(
    image_np,
    gt_display_mask,
    pred_display_masks,
    pred_checkpoint_names,
    eval_task_name,
    sample_idx,
    save_path,
    num_display_classes,
):
    ncols = 2 + len(pred_display_masks)

    fig_width = max(5 * ncols, 12)
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, 5))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, wspace=0.02)

    if ncols == 1:
        axes = [axes]

    gt_overlay = _overlay_mask_on_image(
        image_np,
        gt_display_mask,
        num_display_classes=num_display_classes,
    )

    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    for col_idx, (pred_mask, checkpoint_name) in enumerate(
        zip(pred_display_masks, pred_checkpoint_names),
        start=2,
    ):
        pred_overlay = _overlay_mask_on_image(
            image_np,
            pred_mask,
            num_display_classes=num_display_classes,
        )

        axes[col_idx].imshow(pred_overlay)
        axes[col_idx].set_title(f"Pred after\n{checkpoint_name}")
        axes[col_idx].axis("off")

    fig.suptitle(
        f"Task: {eval_task_name} | Sample: {sample_idx}",
        fontsize=14,
    )

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _evaluate(
    eval_task_name,
    sample_indices,
    train_datasets,
    test_datasets,
    task_names,
    cumulative_class_values_by_task,
    global_class_values,
    device,
    save_root,
):
    """
    For one eval task:
    - pick fixed input images from that task
    - loop over checkpoints from that task onward
    - save one trajectory visualization per input image

    Output format:
    Image | Ground Truth | Pred after checkpoint_i | Pred after checkpoint_i+1 | ...
    """
    eval_task_idx = task_names.index(eval_task_name)

    checkpoint_task_names = task_names[eval_task_idx:]

    save_dir = save_root / _safe_name(eval_task_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_cache = {}

    for sample_idx in sample_indices:
        sample_cache[sample_idx] = {
            "image_np": None,
            "gt_display_mask": None,
            "pred_display_masks": [],
            "pred_checkpoint_names": [],
        }

    for checkpoint_task_name in checkpoint_task_names:
        checkpoint_idx = task_names.index(checkpoint_task_name)
        checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_task_name}.pt"

        checkpoint_class_values = cumulative_class_values_by_task[checkpoint_idx]
        num_classes = len(checkpoint_class_values)

        print(
            f"Evaluating task [{eval_task_name}] "
            f"using checkpoint [{checkpoint_task_name}] "
            f"with {num_classes} classes"
        )

        model = _build_model(num_classes=num_classes, device=device)
        loaded_count = _load_previous_weights(model, checkpoint_path)
        print(f"Loaded {loaded_count} weights from {checkpoint_path.name}")

        eval_label_mapping = _build_label_mapping(train_datasets[eval_task_name])

        eval_dataset = Dataset(
            metadata=test_datasets[eval_task_name],
            label_mapping=eval_label_mapping,
            class_values=checkpoint_class_values,
            augmentation=get_validation_augmentation(seed=SEED + checkpoint_idx),
        )

        for sample_idx in sample_indices:
            image, gt_mask = eval_dataset[sample_idx]

            image = _ensure_image_tensor(image)
            gt_mask = _ensure_mask_tensor(gt_mask)

            pred_mask = _predict_single_image(
                model=model,
                image_tensor=image,
                device=device,
            )

            image_np = _tensor_to_image(image)

            gt_display_mask = _local_index_mask_to_display_mask(
                index_mask=gt_mask,
                local_class_values=checkpoint_class_values,
                global_class_values=global_class_values,
            )

            pred_display_mask = _local_index_mask_to_display_mask(
                index_mask=pred_mask,
                local_class_values=checkpoint_class_values,
                global_class_values=global_class_values,
            )

            if sample_cache[sample_idx]["image_np"] is None:
                sample_cache[sample_idx]["image_np"] = image_np

            if sample_cache[sample_idx]["gt_display_mask"] is None:
                sample_cache[sample_idx]["gt_display_mask"] = gt_display_mask

            sample_cache[sample_idx]["pred_display_masks"].append(pred_display_mask)
            sample_cache[sample_idx]["pred_checkpoint_names"].append(checkpoint_task_name)

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for sample_idx, item in sample_cache.items():
        save_path = save_dir / f"sample_{sample_idx:04d}_trajectory.png"

        _save_trajectory_figure(
            image_np=item["image_np"],
            gt_display_mask=item["gt_display_mask"],
            pred_display_masks=item["pred_display_masks"],
            pred_checkpoint_names=item["pred_checkpoint_names"],
            eval_task_name=eval_task_name,
            sample_idx=sample_idx,
            save_path=save_path,
            num_display_classes=len(global_class_values),
        )

        print(f"Saved visualization: {save_path}")


def main():
    train_datasets = load_task_datasets(
        split="train",
        require_existing_files=True,
    )

    test_datasets = load_task_datasets(
        split="test",
        require_existing_files=True,
    )

    task_names = list(train_datasets.keys())

    cumulative_class_values_by_task, global_class_values = _build_cumulative_class_values(
        train_datasets=train_datasets,
        task_names=task_names,
    )

    print("Task order:")
    for idx, task in enumerate(task_names):
        print(
            f"{idx + 1}. {task} "
            f"| cumulative classes = {cumulative_class_values_by_task[idx]}"
        )

    print(f"Global display classes: {global_class_values}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for eval_task_name in task_names:
        num_available_samples = len(test_datasets[eval_task_name])
        num_samples = min(NUM_SAMPLES_PER_TASK, num_available_samples)

        sample_indices = list(range(num_samples))

        print("=" * 80)
        print(f"Visualizing eval task: {eval_task_name}")
        print(f"Sample indices: {sample_indices}")

        _evaluate(
            eval_task_name=eval_task_name,
            sample_indices=sample_indices,
            train_datasets=train_datasets,
            test_datasets=test_datasets,
            task_names=task_names,
            cumulative_class_values_by_task=cumulative_class_values_by_task,
            global_class_values=global_class_values,
            device=device,
            save_root=VISUALIZATION_DIR,
        )


if __name__ == "__main__":
    main()
import random

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler


class SegmentationReplayBuffer:
    """
    Experience Replay buffer for segmentation.

    Stores:
    - examples: image tensors [3, H, W]
    - labels: mask tensors [H, W]

    Uses reservoir sampling to maintain a fixed-size memory.
    """

    def __init__(self, buffer_size: int):
        self.buffer_size = int(buffer_size)

        self.examples = []
        self.labels = []

        self.num_seen = 0

    def __len__(self):
        return len(self.examples)

    def is_empty(self):
        return len(self.examples) == 0

    @torch.no_grad()
    def add_data(self, examples, labels):
        """
        examples: [B, 3, H, W]
        labels:   [B, H, W]
        """

        if self.buffer_size <= 0:
            return

        examples = examples.detach().cpu()
        labels = labels.detach().cpu().long()

        batch_size = examples.shape[0]

        for i in range(batch_size):
            self.num_seen += 1

            example_i = examples[i].clone()
            label_i = labels[i].clone()

            if len(self.examples) < self.buffer_size:
                self.examples.append(example_i)
                self.labels.append(label_i)
            else:
                replace_idx = random.randint(0, self.num_seen - 1)

                if replace_idx < self.buffer_size:
                    self.examples[replace_idx] = example_i
                    self.labels[replace_idx] = label_i

    def get_data(self, batch_size: int, device):
        """
        Returns:
        - examples: [B, 3, H, W]
        - labels:   [B, H, W]
        """

        if self.is_empty():
            raise RuntimeError("Cannot sample from an empty replay buffer.")

        batch_size = min(batch_size, len(self.examples))
        indices = random.sample(range(len(self.examples)), batch_size)

        examples = torch.stack([self.examples[i] for i in indices]).to(device)
        labels = torch.stack([self.labels[i] for i in indices]).long().to(device)

        return examples, labels

    def state_dict(self):
        return {
            "buffer_size": self.buffer_size,
            "num_seen": self.num_seen,
            "examples": self.examples,
            "labels": self.labels,
        }

    def load_state_dict(self, state):
        self.num_seen = int(state.get("num_seen", 0))

        self.examples = state.get("examples", [])
        self.labels = state.get("labels", [])

        if len(self.examples) > self.buffer_size:
            self.examples = self.examples[-self.buffer_size:]
            self.labels = self.labels[-self.buffer_size:]


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        arch,
        encoder_name,
        in_channels,
        out_classes,
        er_enabled: bool = False,
        er_buffer_size: int = 0,
        er_minibatch_size: int = 8,
        **kwargs,
    ):
        super().__init__()

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            encoder_weights="imagenet",
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)

        self.number_of_classes = out_classes

        self.register_buffer(
            "std",
            torch.tensor(params["std"]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "mean",
            torch.tensor(params["mean"]).view(1, 3, 1, 1),
        )

        # Multi-class segmentation loss
        self.loss_fn = smp.losses.DiceLoss(
            smp.losses.MULTICLASS_MODE,
            from_logits=True,
        )

        # Metric tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # ER settings
        self.er_enabled = er_enabled and er_buffer_size > 0
        self.er_buffer_size = er_buffer_size
        self.er_minibatch_size = er_minibatch_size

        if self.er_enabled:
            self.replay_buffer = SegmentationReplayBuffer(
                buffer_size=er_buffer_size,
            )
        else:
            self.replay_buffer = None

    def forward(self, image):
        image = image.float()

        mean = self.mean.to(device=image.device, dtype=image.dtype)
        std = self.std.to(device=image.device, dtype=image.dtype)

        image = (image - mean) / std

        mask = self.model(image)

        return mask

    def _compute_segmentation_metrics(self, logits_mask, mask):
        prob_mask = logits_mask.softmax(dim=1)
        pred_mask = prob_mask.argmax(dim=1)

        pred_mask_cpu = pred_mask.detach().cpu()
        mask_cpu = mask.detach().cpu()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask_cpu,
            mask_cpu,
            mode="multiclass",
            num_classes=self.number_of_classes,
        )

        return tp, fp, fn, tn

    def shared_step(self, batch, stage):
        image, mask = batch

        assert image.ndim == 4
        assert mask.ndim == 3

        mask = mask.long().to(image.device)

        logits_mask = self.forward(image)
        logits_mask = logits_mask.contiguous()

        assert logits_mask.shape[1] == self.number_of_classes

        loss = self.loss_fn(logits_mask, mask)

        tp, fp, fn, tn = self._compute_segmentation_metrics(
            logits_mask,
            mask,
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def training_step(self, batch, batch_idx):
        image, mask = batch

        assert image.ndim == 4
        assert mask.ndim == 3

        real_batch_size = image.shape[0]

        mask = mask.long().to(image.device)

        train_inputs = image
        train_labels = mask

        # Experience Replay
        if self.er_enabled and not self.replay_buffer.is_empty():
            buf_inputs, buf_labels = self.replay_buffer.get_data(
                batch_size=self.er_minibatch_size,
                device=self.device,
            )

            train_inputs = torch.cat([train_inputs, buf_inputs], dim=0)
            train_labels = torch.cat([train_labels, buf_labels], dim=0)

        logits_mask = self.forward(train_inputs)
        logits_mask = logits_mask.contiguous()

        assert logits_mask.shape[1] == self.number_of_classes

        loss = self.loss_fn(logits_mask, train_labels)

        # Compute metrics only on the current batch, not replay samples
        current_logits = logits_mask[:real_batch_size]
        current_labels = train_labels[:real_batch_size]

        tp, fp, fn, tn = self._compute_segmentation_metrics(
            current_logits,
            current_labels,
        )

        self.training_step_outputs.append(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=real_batch_size,
        )

        if self.er_enabled:
            self.replay_buffer.add_data(
                examples=image.detach(),
                labels=mask.detach(),
            )

            self.log(
                "replay_buffer_size",
                float(len(self.replay_buffer)),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=real_batch_size,
            )

        return loss

    def shared_epoch_end(self, outputs, stage):
        if len(outputs) == 0:
            return

        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(
            tp,
            fp,
            fn,
            tn,
            reduction="micro-imagewise",
        )

        dataset_iou = smp.metrics.iou_score(
            tp,
            fp,
            fn,
            tn,
            reduction="micro",
        )

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        image, _ = batch

        valid_info = self.shared_step(batch, "valid")

        self.validation_step_outputs.append(
            {
                "tp": valid_info["tp"],
                "fp": valid_info["fp"],
                "fn": valid_info["fn"],
                "tn": valid_info["tn"],
            }
        )

        self.log(
            "valid_loss",
            valid_info["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=image.shape[0],
        )

        return valid_info["loss"]

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        image, _ = batch

        test_info = self.shared_step(batch, "test")

        self.test_step_outputs.append(
            {
                "tp": test_info["tp"],
                "fp": test_info["fp"],
                "fn": test_info["fn"],
                "tn": test_info["tn"],
            }
        )

        self.log(
            "test_loss",
            test_info["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=image.shape[0],
        )

        return test_info["loss"]

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def get_er_buffer_state(self):
        if not self.er_enabled or self.replay_buffer is None:
            return None

        return self.replay_buffer.state_dict()

    def load_er_buffer_state(self, state):
        if state is None:
            return

        if self.replay_buffer is None:
            self.replay_buffer = SegmentationReplayBuffer(
                buffer_size=self.er_buffer_size,
            )

        self.replay_buffer.load_state_dict(state)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=2e-4,
        )

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

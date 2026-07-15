import random
from typing import Optional

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler


class SegmentationReplayBuffer:
    """
    Replay buffer for segmentation DER++.

    Stores:
    - examples: image tensors [3, H, W]
    - labels: mask tensors [H, W]
    - logits: old model logits [C_old, H, W]

    Reservoir sampling is used so the buffer remains fixed-size.
    """

    def __init__(
        self,
        buffer_size: int,
        logits_dtype: torch.dtype = torch.float16,
    ):
        self.buffer_size = int(buffer_size)
        self.logits_dtype = logits_dtype

        self.examples = []
        self.labels = []
        self.logits = []

        self.num_seen = 0

    def __len__(self):
        return len(self.examples)

    def is_empty(self):
        return len(self.examples) == 0

    @torch.no_grad()
    def add_data(self, examples, labels, logits):
        """
        examples: [B, 3, H, W]
        labels:   [B, H, W]
        logits:   [B, C, H, W]
        """

        if self.buffer_size <= 0:
            return

        examples = examples.detach().cpu()
        labels = labels.detach().cpu().long()
        logits = logits.detach().cpu().to(dtype=self.logits_dtype)

        batch_size = examples.shape[0]

        for i in range(batch_size):
            self.num_seen += 1

            example_i = examples[i].clone()
            label_i = labels[i].clone()
            logit_i = logits[i].clone()

            if len(self.examples) < self.buffer_size:
                self.examples.append(example_i)
                self.labels.append(label_i)
                self.logits.append(logit_i)
            else:
                replace_idx = random.randint(0, self.num_seen - 1)

                if replace_idx < self.buffer_size:
                    self.examples[replace_idx] = example_i
                    self.labels[replace_idx] = label_i
                    self.logits[replace_idx] = logit_i

    def _sample_indices(self, batch_size: int):
        batch_size = min(batch_size, len(self.examples))
        return random.sample(range(len(self.examples)), batch_size)

    def get_data(self, batch_size: int, device, with_logits: bool = True):
        """
        Returns sampled buffer data.

        If with_logits=True:
            returns examples, labels, padded_logits, valid_logit_mask

        valid_logit_mask is needed because old tasks may have fewer classes
        than the current model.
        """

        if self.is_empty():
            raise RuntimeError("Cannot sample from an empty replay buffer.")

        indices = self._sample_indices(batch_size)

        examples = torch.stack([self.examples[i] for i in indices]).to(device)
        labels = torch.stack([self.labels[i] for i in indices]).long().to(device)

        if not with_logits:
            return examples, labels

        logits_list = [self.logits[i] for i in indices]

        max_channels = max(logit.shape[0] for logit in logits_list)
        height = logits_list[0].shape[-2]
        width = logits_list[0].shape[-1]

        padded_logits = torch.zeros(
            len(indices),
            max_channels,
            height,
            width,
            dtype=logits_list[0].dtype,
        )

        valid_logit_mask = torch.zeros(
            len(indices),
            max_channels,
            1,
            1,
            dtype=torch.float32,
        )

        for i, logit in enumerate(logits_list):
            channels = logit.shape[0]
            padded_logits[i, :channels] = logit
            valid_logit_mask[i, :channels] = 1.0

        padded_logits = padded_logits.to(device)
        valid_logit_mask = valid_logit_mask.to(device)

        return examples, labels, padded_logits, valid_logit_mask

    def state_dict(self):
        return {
            "buffer_size": self.buffer_size,
            "num_seen": self.num_seen,
            "examples": self.examples,
            "labels": self.labels,
            "logits": self.logits,
        }

    def load_state_dict(self, state):
        self.num_seen = int(state.get("num_seen", 0))

        self.examples = state.get("examples", [])
        self.labels = state.get("labels", [])
        self.logits = state.get("logits", [])

        if len(self.examples) > self.buffer_size:
            self.examples = self.examples[-self.buffer_size:]
            self.labels = self.labels[-self.buffer_size:]
            self.logits = self.logits[-self.buffer_size:]


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        arch,
        encoder_name,
        in_channels,
        out_classes,
        derpp_enabled: bool = False,
        derpp_buffer_size: int = 0,
        derpp_minibatch_size: int = 8,
        derpp_alpha: float = 0.5,
        derpp_beta: float = 1.0,
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

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.number_of_classes = out_classes

        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.loss_fn = smp.losses.DiceLoss(
            smp.losses.MULTICLASS_MODE,
            from_logits=True,
        )

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # DER++ settings
        self.derpp_enabled = derpp_enabled and derpp_buffer_size > 0
        self.derpp_buffer_size = derpp_buffer_size
        self.derpp_minibatch_size = derpp_minibatch_size
        self.derpp_alpha = derpp_alpha
        self.derpp_beta = derpp_beta

        if self.derpp_enabled:
            self.replay_buffer = SegmentationReplayBuffer(
                buffer_size=derpp_buffer_size,
                logits_dtype=torch.float16,
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

    def shared_step(self, batch, stage, return_logits: bool = False):
        image, mask = batch

        assert image.ndim == 4
        assert mask.ndim == 3

        mask = mask.long().to(image.device)

        logits_mask = self.forward(image)
        logits_mask = logits_mask.contiguous()

        assert logits_mask.shape[1] == self.number_of_classes

        loss = self.loss_fn(logits_mask, mask)

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

        output = {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        if return_logits:
            output["logits"] = logits_mask.detach()

        return output

    def _derpp_mse_loss(self):
        """
        Dark Experience Replay loss.

        Matches current logits with stored old logits.
        Handles class expansion by comparing only valid old channels.
        """

        buf_inputs, _, buf_logits, valid_logit_mask = self.replay_buffer.get_data(
            batch_size=self.derpp_minibatch_size,
            device=self.device,
            with_logits=True,
        )

        buf_outputs = self.forward(buf_inputs)

        old_channels = buf_logits.shape[1]

        if buf_outputs.shape[1] < old_channels:
            raise RuntimeError(
                f"Current model has {buf_outputs.shape[1]} classes, "
                f"but buffer logits have {old_channels} classes."
            )

        buf_outputs = buf_outputs[:, :old_channels]

        diff = buf_outputs - buf_logits.float()
        diff = diff * valid_logit_mask

        denom = valid_logit_mask.sum() * buf_logits.shape[-2] * buf_logits.shape[-1]
        denom = denom.clamp_min(1.0)

        mse_loss = diff.pow(2).sum() / denom

        return mse_loss

    def _derpp_label_loss(self):
        """
        Experience Replay supervised loss.
        """

        buf_inputs, buf_labels = self.replay_buffer.get_data(
            batch_size=self.derpp_minibatch_size,
            device=self.device,
            with_logits=False,
        )

        buf_outputs = self.forward(buf_inputs)
        buf_outputs = buf_outputs.contiguous()

        replay_loss = self.loss_fn(buf_outputs, buf_labels.long())

        return replay_loss

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

    def training_step(self, batch, batch_idx):
        image, mask = batch

        train_info = self.shared_step(
            batch,
            stage="train",
            return_logits=True,
        )

        base_loss = train_info["loss"]
        total_loss = base_loss

        if self.derpp_enabled and not self.replay_buffer.is_empty():
            derpp_mse_loss = self._derpp_mse_loss()
            derpp_label_loss = self._derpp_label_loss()

            total_loss = (
                total_loss
                + self.derpp_alpha * derpp_mse_loss
                + self.derpp_beta * derpp_label_loss
            )

            self.log(
                "train_derpp_mse_loss",
                derpp_mse_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=image.shape[0],
            )

            self.log(
                "train_derpp_label_loss",
                derpp_label_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                batch_size=image.shape[0],
            )

        self.log(
            "train_base_loss",
            base_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            batch_size=image.shape[0],
        )

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=image.shape[0],
        )

        if self.derpp_enabled:
            self.replay_buffer.add_data(
                examples=image.detach(),
                labels=mask.detach(),
                logits=train_info["logits"],
            )

            self.log(
                "replay_buffer_size",
                float(len(self.replay_buffer)),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=image.shape[0],
            )

        self.training_step_outputs.append(
            {
                "tp": train_info["tp"],
                "fp": train_info["fp"],
                "fn": train_info["fn"],
                "tn": train_info["tn"],
            }
        )

        return total_loss

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        image, _ = batch

        valid_info = self.shared_step(batch, "valid")

        self.log(
            "valid_loss",
            valid_info["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=image.shape[0],
        )

        self.validation_step_outputs.append(
            {
                "tp": valid_info["tp"],
                "fp": valid_info["fp"],
                "fn": valid_info["fn"],
                "tn": valid_info["tn"],
            }
        )

        return valid_info["loss"]

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        image, _ = batch

        test_info = self.shared_step(batch, "test")

        self.log(
            "test_loss",
            test_info["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=image.shape[0],
        )

        self.test_step_outputs.append(
            {
                "tp": test_info["tp"],
                "fp": test_info["fp"],
                "fn": test_info["fn"],
                "tn": test_info["tn"],
            }
        )

        return test_info["loss"]

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def get_derpp_buffer_state(self):
        if not self.derpp_enabled or self.replay_buffer is None:
            return None

        return self.replay_buffer.state_dict()

    def load_derpp_buffer_state(self, state):
        if state is None:
            return

        if self.replay_buffer is None:
            self.replay_buffer = SegmentationReplayBuffer(
                buffer_size=self.derpp_buffer_size,
                logits_dtype=torch.float16,
            )

        self.replay_buffer.load_state_dict(state)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)

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

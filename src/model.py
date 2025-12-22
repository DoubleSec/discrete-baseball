"""Primarily for the LightningModule that contains the model.

Other network modules, etc. can go here or not, depending on how confusing it is.
"""

from typing import Any

import torch
import lightning.pytorch as pl
from torchmetrics.text import Perplexity
from torchmetrics.classification import BinaryAUROC, MulticlassAUROC, MulticlassRecall
from x_transformers import TransformerWrapper, Decoder

from .processing import Stringifier, TimeTokenizer


class AutoregressivePretrainedModel(pl.LightningModule):
    """Model for whatever we're doing."""

    def __init__(
        self,
        optimizer_params: dict[str, Any],
        # Data artifacts
        stringifiers: list[Stringifier],
        time_tokenizer: TimeTokenizer,
        complete_vocab: dict[str:int],
        # Model parameters
        d_model: int,
        max_seq_len: int,
        depth: int,
        heads: int,
        emb_dropout: float,
        attn_dropout: float,
    ):
        """Initialize the model.

        - optimizer_params: dictionary of parameters to initialize optimizer with.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer_params = optimizer_params
        self.ignore_index = complete_vocab["<PAD>"]

        self.transformer = TransformerWrapper(
            num_tokens=len(complete_vocab),
            max_seq_len=max_seq_len,
            return_only_embed=True,
            emb_dropout=emb_dropout,
            attn_layers=Decoder(
                dim=d_model,
                depth=depth,
                heads=heads,
                rotary_pos_emb=True,
                attn_flash=True,
                attn_dropout=attn_dropout,
            ),
        )
        self.prediction_head = torch.nn.Linear(d_model, len(complete_vocab))

        # Metrics
        self.train_perplexity = Perplexity(ignore_index=self.ignore_index)
        self.valid_perplexity = Perplexity(ignore_index=self.ignore_index)

    def configure_optimizers(self):
        """Lightning hook for optimizer setup.

        Body here is just an example, although it probably works okay for a lot of things.
        We can't pass arguments to this directly, so they need to go to the init.
        """

        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        return optimizer

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for the model.

        - x is compatible with the output of data.TrainingDataset.__getitem__, literally the output
          from a dataloader.

        Returns whatever the output of the model is.
        """

        return self.transformer(x)

    def step(self, stage: str, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generic step for training or validation, assuming they're similar.

        - stage: one of "train" or "valid"
        - x: dictionary of torch tensors, input to model, targets, etc.

        This MUST log "valid_loss" during the validation step in order to have the model checkpointing work as written.

        Returns loss as one-element tensor.
        """

        # Strip off EOS: (n, s-1)
        input = x["x"][:, :-1]
        # Strip off SOS: (n, s-1)
        targets = x["x"][:, 1:]

        # (n, s-1, e)
        embeddings = self.transformer(input)
        # (n, v, s-1)
        logits = self.prediction_head(embeddings)

        loss = torch.nn.functional.cross_entropy(
            logits.mT, targets, ignore_index=self.ignore_index, reduction="mean"
        )

        # Update the metric
        if stage == "train":
            self.train_perplexity(logits, targets)
            metric = self.train_perplexity
        elif stage == "valid":
            self.valid_perplexity(logits, targets)
            metric = self.valid_perplexity

        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_perplexity", metric)

        return loss

    def training_step(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lightning hook for training."""
        return self.step("train", x)

    def validation_step(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lightning hook for validation."""
        return self.step("valid", x)


class PitcherPredictionPretrainedModel(pl.LightningModule):
    """Model for whatever we're doing."""

    def __init__(
        self,
        optimizer_params: dict[str, Any],
        # Data artifacts
        stringifiers: list[Stringifier],
        time_tokenizer: TimeTokenizer,
        complete_vocab: dict[str:int],
        # Model parameters
        d_model: int,
        max_seq_len: int,
        depth: int,
        heads: int,
        emb_dropout: float,
        attn_dropout: float,
        n_classes: int,
    ):
        """Initialize the model.

        Expects targets as "pitcher_id"

        - optimizer_params: dictionary of parameters to initialize optimizer with.
        """

        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer_params = optimizer_params
        self.ignore_index = complete_vocab["<PAD>"]

        self.transformer = TransformerWrapper(
            num_tokens=len(complete_vocab),
            max_seq_len=max_seq_len,
            return_only_embed=True,
            emb_dropout=emb_dropout,
            attn_layers=Decoder(
                dim=d_model,
                depth=depth,
                heads=heads,
                rotary_pos_emb=True,
                attn_flash=True,
                attn_dropout=attn_dropout,
            ),
        )
        self.prediction_head = torch.nn.Linear(d_model, n_classes)

        # Metrics (lol make this a metric dict)
        # TKTK Lightning isn't managing these correctly anyway.
        self.train_auroc = MulticlassAUROC(n_classes, average="weighted")
        self.valid_auroc = MulticlassAUROC(n_classes, average="weighted")
        self.train_recall_1 = MulticlassRecall(n_classes, average="micro", top_k=1)
        self.valid_recall_1 = MulticlassRecall(n_classes, average="micro", top_k=1)
        self.train_recall_5 = MulticlassRecall(n_classes, average="micro", top_k=5)
        self.valid_recall_5 = MulticlassRecall(n_classes, average="micro", top_k=5)

    def configure_optimizers(self):
        """Lightning hook for optimizer setup.

        Body here is just an example, although it probably works okay for a lot of things.
        We can't pass arguments to this directly, so they need to go to the init.
        """

        optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        return optimizer

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass for the model.

        - x is compatible with the output of data.TrainingDataset.__getitem__, literally the output
          from a dataloader.

        Returns whatever the output of the model is.
        """
        n = x.shape[0]
        return self.transformer(x)[torch.arange(n, device=x.device), idx, :]

    def step(self, stage: str, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generic step for training or validation, assuming they're similar.

        - stage: one of "train" or "valid"
        - x: dictionary of torch tensors, input to model, targets, etc.

        This MUST log "valid_loss" during the validation step in order to have the model checkpointing work as written.

        Returns loss as one-element tensor.
        """

        input = x["x"]
        idx = x["sequence_length"] - 1
        targets = x["pitcher_id"]
        n = idx.shape[0]

        # (n, s, e)
        embeddings = self.transformer(input)
        # (n, e)
        last_embeddings = embeddings[torch.arange(n).to(idx), idx, :]
        # (n, n_classes)
        logits = self.prediction_head(last_embeddings)

        loss = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")

        # Update the metric
        if stage == "train":
            self.train_auroc(logits, targets)
            auroc = self.train_auroc
            self.train_recall_1(logits, targets)
            recall_1 = self.train_recall_1
            self.train_recall_5(logits, targets)
            recall_5 = self.train_recall_5
        elif stage == "valid":
            self.valid_auroc(logits, targets)
            auroc = self.valid_auroc
            self.valid_recall_1(logits, targets)
            recall_1 = self.valid_recall_1
            self.valid_recall_5(logits, targets)
            recall_5 = self.valid_recall_5

        self.log(f"{stage}_loss", loss)
        self.log(f"{stage}_auroc", auroc)
        self.log(f"{stage}_recall@1", recall_1)
        self.log(f"{stage}_recall@5", recall_5)

        return loss

    def training_step(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lightning hook for training."""
        return self.step("train", x)

    def validation_step(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lightning hook for validation."""
        return self.step("valid", x)


class OutPredictor(pl.LightningModule):

    def __init__(
        self,
        checkpoint_path,
        optimizer_params,
        n_clip_from_end: int,
        unfreeze_all: bool = False,
    ):
        """Initialize.

        n_clip_from_end allows us to not include output from the last description or the EOS token
        for each sequence, because those leak the target.
        """

        super().__init__()
        self.optimizer_params = optimizer_params
        self.unfreeze_all = unfreeze_all

        self.transformer = AutoregressivePretrainedModel.load_from_checkpoint(
            checkpoint_path,
            weights_only=False,
        )
        # Remove the pretraining head.
        del self.transformer.prediction_head
        self.transformer.train()

        self.out_predictor = torch.nn.Linear(self.transformer.hparams["d_model"], 1)
        self.pad_idx = self.transformer.ignore_index
        self.n_clip_from_end = n_clip_from_end

        # Metrics
        self.train_auroc = BinaryAUROC(ignore_index=2)
        self.valid_auroc = BinaryAUROC(ignore_index=2)

    def configure_optimizers(self):

        # Only the predictor head.
        if self.unfreeze_all is True:
            optimizer = torch.optim.AdamW(self.parameters(), **self.optimizer_params)
        else:
            optimizer = torch.optim.AdamW(
                self.out_predictor.parameters(), **self.optimizer_params
            )
        return optimizer

    def step(self, stage: str, x: dict[str, Any]):

        n, s = x["x"].shape

        # (n, s)
        target = x["is_out"][:, None].expand(-1, s)

        # (n, s)
        target_mask = (
            torch.arange(s, device=target.device)[None, :].expand(n, -1)
            < (x["sequence_length"].unsqueeze(-1) - self.n_clip_from_end)
        ).float()

        # (n, s, k)
        embeddings = self.transformer(x["x"])
        # (n, s)
        predictions = self.out_predictor(embeddings).squeeze(-1)
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                predictions, target, reduction="none"
            )
            * target_mask
        ).sum() / target_mask.sum()

        self.log(f"{stage}_loss", loss)

        metric = self.train_auroc if stage == "train" else self.valid_auroc

        target = target.int()
        metric_target = torch.where(
            target_mask == 1, target, torch.ones([]).to(target) * 2
        )
        metric(predictions, metric_target)
        self.log(f"{stage}_auroc", metric)

        return loss

    def forward(self, x: dict[str, Any]):

        # (n, s, k)
        embeddings = self.transformer(x["x"])
        # (n, s)
        return self.out_predictor(embeddings).squeeze(-1)

    def training_step(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lightning hook for training."""
        return self.step("train", x)

    def validation_step(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lightning hook for validation."""
        return self.step("valid", x)

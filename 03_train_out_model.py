"""Train a model."""

import yaml
import torch
from lightning import pytorch as pl
import polars

from src.data import TrainingDataset, EVENT_IS_OUT
from src.model import AutoregressivePretrainedModel, OutPredictor


def construct_targets(df, keys):

    return df.with_columns(
        polars.col("events")
        .replace_strict(EVENT_IS_OUT, return_dtype=polars.Float32, default=0)
        .max()
        .over(partition_by=keys)
        .alias("is_out")
    )


if __name__ == "__main__":

    # Read the config ---------------------------------------
    # (Probably should use hydra eventually.)

    with open("cfg/out_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    # Hashtag reproducibility
    torch.manual_seed(config["random_seed"])

    # Generic dataset preparation ---------------------------

    pretrained_hparams = AutoregressivePretrainedModel.load_from_checkpoint(
        config["pretrained_path"],
        weights_only=False,
    ).hparams

    ds = TrainingDataset(
        path=config["prepared_data_path"],
        **config["dataset_params"],
        state_dict={
            "stringifiers": pretrained_hparams["stringifiers"],
            "time_tokenizer": pretrained_hparams["time_tokenizer"],
            "complete_vocab": pretrained_hparams["complete_vocab"],
        },
        extra_processing=construct_targets,
        targets=config["targets"],
    )
    print(f"Number of at-bats: {len(ds)}")

    train_ds, valid_ds, test_ds = torch.utils.data.random_split(ds, [0.75, 0.15, 0.1])

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        shuffle=True,
        **config["dataloader_params"],
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        **config["dataloader_params"],
    )
    # For the moment we're not doing anything with the test dataset.

    # Set up the trainer ------------------------------------

    trainer = pl.Trainer(
        **config["trainer_params"],
        logger=pl.loggers.TensorBoardLogger("./outs"),
        callbacks=pl.callbacks.ModelCheckpoint(
            dirpath="./out_model",
            save_top_k=1,
            monitor="valid_loss",
            filename="{epoch}-{valid_loss:.4f}",
        ),
    )

    with trainer.init_module():

        net = OutPredictor(
            config["pretrained_path"],
            **config["model_params"],
            optimizer_params=config["optimizer_params"],
        )
        # Maybe
        # net.compile()

    torch.set_float32_matmul_precision("medium")
    trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=valid_dl)

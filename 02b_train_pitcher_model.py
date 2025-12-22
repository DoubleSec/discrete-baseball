"""Train a model."""

import yaml
import torch
from lightning import pytorch as pl
import polars

from src.data import TrainingDataset
from src.model import PitcherPredictionPretrainedModel


def construct_targets(df, keys):

    # Er, we need to do this first to get the correct set of pitchers.
    df = df.filter(polars.col("game_type").is_in(["R", "F", "D", "L", "W"]))

    pitcher_map = {
        pname: i for i, pname in enumerate(sorted(df["player_name"].unique().to_list()))
    }

    # Er, we have to do this filter an extra time.
    return df.with_columns(
        polars.col("player_name")
        .replace_strict(pitcher_map, return_dtype=polars.Int32, default=0)
        .alias("pitcher_id")
    )


if __name__ == "__main__":

    # Read the config ---------------------------------------
    # (Probably should use hydra eventually.)

    with open("cfg/pitcher_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    # Hashtag reproducibility
    torch.manual_seed(config["random_seed"])

    # Generic dataset preparation ---------------------------

    ds = TrainingDataset(
        path=config["prepared_data_path"],
        **config["dataset_params"],
        extra_processing=construct_targets,
    )
    print(f"Number of at-bats: {len(ds)}")
    print(f"Number of pitchers: {ds.df['pitcher_id'].max() + 1}")
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
        logger=pl.loggers.TensorBoardLogger("./pretrained_pp"),
        callbacks=pl.callbacks.ModelCheckpoint(
            dirpath="./model",
            save_top_k=1,
            monitor="valid_loss",
            filename="{epoch}-{valid_loss:.4f}",
        ),
    )

    with trainer.init_module():

        net = PitcherPredictionPretrainedModel(
            **ds.get_state(),
            **config["model_params"],
            optimizer_params=config["optimizer_params"],
            n_classes=ds.df["pitcher_id"].max() + 1,
        )
        # Maybe
        # net.compile()

    torch.set_float32_matmul_precision("medium")
    trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=valid_dl)

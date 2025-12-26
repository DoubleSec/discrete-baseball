"""Train a model."""

import yaml
import torch
from lightning import pytorch as pl

from src.data import TrainingDataset
from src.model import AutoregressivePretrainedModel

ONE_TRILLION = 1_000_000_000_000

if __name__ == "__main__":

    # Read the config ---------------------------------------
    # (Probably should use hydra eventually.)

    with open("cfg/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    # Hashtag reproducibility
    torch.manual_seed(config["random_seed"])

    # Generic dataset preparation ---------------------------

    ds = TrainingDataset(
        path=config["prepared_data_path"],
        **config["dataset_params"],
    )
    train_ds, valid_ds, test_ds = torch.utils.data.random_split(
        ds, config["data_proportions"]
    )

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
        logger=pl.loggers.TensorBoardLogger("./pretrained"),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="./model",
                save_top_k=1,
                monitor="valid_loss",
                filename="{epoch}-{valid_loss:.4f}",
            ),
            pl.callbacks.LearningRateMonitor(),
        ],
    )

    # This section is purely experimentation and very, very optional. ---------

    # How many FLOPs for forward/backward
    with torch.device("meta"):
        net = AutoregressivePretrainedModel(
            **ds.get_state(),
            **config["model_params"],
            optimizer_params=config["optimizer_params"],
        )
        x = torch.randint(0, 50, [1, ds.max_length - 1])
        y = torch.randint(0, 50, [1, ds.max_length - 1])

    n_params = sum(p.numel() for p in net.parameters())

    forward_and_backwards_flops = pl.utilities.measure_flops(
        net, lambda: net(x), lambda z: net.loss(z, y)
    )
    teraflops_per_training_obs = forward_and_backwards_flops / ONE_TRILLION
    expected_total_teraflops = (
        teraflops_per_training_obs
        * len(ds)
        * config["data_proportions"][0]  # Train fraction
        * config["trainer_params"]["max_epochs"]
        * ds.data_stats()["average_fill"]  # Only this much is useful per observation.
    )

    print(
        f"Number of training tokens (M): {ds.data_stats()['n_tokens'] * config["data_proportions"][0] * config['trainer_params']['max_epochs'] / 1_000_000 :.3f}"
    )
    print(f"Number of parameters (M): {n_params / 1_000_000 :.3f}")
    print(f"Expected TeraFLOPs: {expected_total_teraflops :.3f}")

    # Train the thing for real.

    total_steps = len(train_dl) * config["trainer_params"]["max_epochs"]
    print(f"Total steps: {total_steps}")

    with trainer.init_module():

        net = AutoregressivePretrainedModel(
            **ds.get_state(),
            **config["model_params"],
            optimizer_params=config["optimizer_params"] | {"total_steps": total_steps},
        )

    torch.set_float32_matmul_precision("medium")
    trainer.fit(net, train_dataloaders=train_dl, val_dataloaders=valid_dl)

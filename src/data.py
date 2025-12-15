"""Contains tools for initial data preprocessing and then data loading during training."""

from pybaseball import statcast
import polars as pl

import torch
from torch.utils.data import Dataset

from logzero import logger


def initial_prep(
    path: str,
    start_date: str,
    end_date: str,
) -> str:
    """Do one-time data preparation.
    Here we'll just download statcast data and save it.
    """

    init_data = statcast(start_dt=start_date, end_dt=end_date)
    init_data = pl.DataFrame(init_data)
    logger.info(f"Writing {len(init_data)} rows to {path}")
    init_data.write_parquet(path)
    return path


class TrainingDataset(Dataset):
    """Torch Dataset class for training."""

    def __init__(self, path: str, *args, **kwargs):
        """Load a dataset and do any prep required.

        If at all possible, this should be deterministic, given a specific file.
        """
        super().__init__()
        raise NotImplementedError

    def __len__(self) -> int:
        """Obvious"""
        raise NotImplementedError

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        """Return a single instance of the training dataset, as a dictionary."""
        raise NotImplementedError

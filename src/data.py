"""Contains tools for initial data preprocessing and then data loading during training."""

from typing import Any

from pybaseball import statcast
import polars as pl
import torch
from torch.utils.data import Dataset
from logzero import logger

from processing import (
    TimeTokenizer,
    Stringifier,
    make_vocabulary,
)


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

    def __init__(
        self,
        path: str,
        keys: list[str],
        order_columns: list[str],
        time_column: str,
        features: dict[str, dict[str, Any]],
        special_tokens: list[str],
        keyword_args: dict[str, Any],
        state_dict: dict[str:Any] | None = None,
    ):
        """Load a dataset and do any prep required.

        If at all possible, this should be deterministic, given a specific file.
        """
        super().__init__()

        if state_dict is None:

            self.keys = keys
            self.order_columns = order_columns
            self.time_column = time_column

            # Prepare the dataset
            df = (
                pl.read_parquet(path)
                .filter(pl.col("game_type").is_in(["R", "F", "D", "L", "W"]))
                .select(
                    *keys,
                    *order_columns,
                    # Non-generalizable gross hack
                    (
                        pl.col("inning")
                        + pl.when(pl.col("inning_topbot") == "Top")
                        .then(0)
                        .otherwise(0.5)
                    ).alias("inning"),
                    *[column for column in features],
                )
                .with_columns(
                    (pl.col(time_column) - pl.col(time_column).shift(1))
                    .over(partition_by=keys, order_by=order_columns)
                    .alias("time_diffs")
                )
            )

            self.time_tokenizer = TimeTokenizer.from_data(df["time_diffs"])

            self.stringifiers = {
                column: Stringifier.from_data(
                    df[column], **col_args, kwargs=keyword_args
                )
                for column, col_args in features.items()
            }

            self.complete_vocab = make_vocabulary(
                self.stringifiers.values(),
                self.time_tokenizer,
                special_tokens=special_tokens,
            )
            print(f"Vocab size: {len(self.complete_vocab)}")

            df = (
                df.select(
                    *keys,
                    *order_columns,
                    self.time_tokenizer.transform(pl.col("time_diffs")).alias(
                        "time_diffs"
                    ),
                    *[
                        s.transform(pl.col(n)).alias(n)
                        for n, s in self.stringifiers.items()
                    ],
                )
                .with_columns(
                    pl.concat_list(
                        "time_diffs", *[pl.col(n) for n in self.stringifiers]
                    )
                    .list.drop_nulls()
                    .alias("feature_list")
                )
                .select(*keys, *order_columns, "feature_list")
                .explode("feature_list")
                .with_columns(
                    pl.col("feature_list")
                    .replace(self.complete_vocab)
                    .cast(pl.Int64)
                    .alias("processed_list"),
                )
            )

            df = (
                df.sort(*order_columns)
                .group_by(*keys)  # Within-group order is always kept
                .agg("processed_list", "feature_list")
            )

            # Add SOS and EOS I guess?
            # Also doesn't generalize
            self.df = df.with_columns(
                pl.concat_list(
                    pl.lit(self.complete_vocab["<SOS>"]),
                    pl.col("processed_list"),
                    pl.lit(self.complete_vocab["<EOS>"]),
                ).alias("processed_list"),
                pl.col("processed_list").list.len().alias("sequence_length"),
            )
            self.max_length = self.df["sequence_length"].max()

        else:
            raise NotImplementedError("TKTK load from saved")

    def __len__(self) -> int:
        """Obvious"""
        return self.df.height

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        """Return a single instance of the training dataset, as a dictionary."""

        row = self.df.row(idx, named=True)
        x = torch.nn.functional.pad(
            torch.tensor(row["processed_list"]),
            (0, self.max_length - row["sequence_length"]),
        )
        return {k: row[k] for k in self.keys} | {"x": x}

    @classmethod
    def from_saved(cls, path, *args, **kwargs):
        raise NotImplementedError("TKTK load from saved")


if __name__ == "__main__":

    import yaml

    with open("./cfg/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    df = TrainingDataset(
        path=config["prepared_data_path"],
        **config["dataset_params"],
    )

    print(len(df))
    print(df[500])

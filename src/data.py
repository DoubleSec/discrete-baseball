"""Contains tools for initial data preprocessing and then data loading during training."""

from typing import Any

from pybaseball import statcast
import polars as pl
import torch
from torch.utils.data import Dataset
from logzero import logger

from .processing import (
    TimeTokenizer,
    Stringifier,
    make_vocabulary,
)

EVENT_IS_OUT = {
    "catcher_interf": 0,
    "double": 0,
    "sac_bunt": 1,
    "triple_play": 1,
    "fielders_choice_out": 1,
    "fielders_choice": 1,
    "hit_by_pitch": 0,
    "truncated_pa": 0,
    "sac_fly": 1,
    "field_out": 1,
    "strikeout": 1,
    "single": 0,
    "double_play": 1,
    "triple": 0,
    "force_out": 1,
    "field_error": 0,
    "grounded_into_double_play": 1,
    "home_run": 0,
    "walk": 0,
    "sac_fly_double_play": 1,
    "None": 0,
    "strikeout_double_play": 1,
    "intent_walk": 0,
    "sac_bunt_double_play": 1,
}


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
        special_tokens: list[str] | None = None,
        features: dict[str, dict[str, Any]] | None = None,
        keyword_args: dict[str, Any] | None = None,
        state_dict: dict[str:Any] | None = None,
        extra_processing: callable | None = None,
        targets: list[str] | None = None,
    ):
        """Load a dataset and do any prep required.

        If at all possible, this should be deterministic, given a specific file.
        """
        super().__init__()

        self.keys = keys
        # The expectation is that targets are usable as such after `extra_preprocessing`.
        targets = [] if targets is None else targets
        self.targets = targets

        if state_dict is not None:
            print("Configuring dataset using state dict.")
            features = state_dict["stringifiers"].keys()

        df = pl.read_parquet(path)
        if extra_processing is not None:
            df = extra_processing(df, keys)

        # Prepare the dataset
        df = (
            df.filter(pl.col("game_type").is_in(["R", "F", "D", "L", "W"]))
            # Here so it's selectable with a stringifier and ordered correctly
            .with_columns(
                # Only have the pitcher name on the first pitch
                pl.when(pl.col("pitch_number") == 1)
                .then(pl.col("player_name"))
                .otherwise(None)
                .alias("pitcher_name"),
            )
            .select(
                *keys,
                *order_columns,
                *targets,
                # Non-generalizable gross hack
                (
                    pl.col("inning")
                    + pl.when(pl.col("inning_topbot") == "Top").then(0).otherwise(0.5)
                ).alias("inning"),
                *[column for column in features],
            )
            .with_columns(
                (pl.col(time_column) - pl.col(time_column).shift(1))
                .over(partition_by=keys, order_by=order_columns)
                .alias("time_diffs")
            )
        )

        if state_dict is None:
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
        else:
            self.time_tokenizer = state_dict["time_tokenizer"]
            self.stringifiers = state_dict["stringifiers"]
            self.complete_vocab = state_dict["complete_vocab"]

        print(f"Vocab size: {len(self.complete_vocab)}")

        df = (
            df.select(
                *keys,
                *targets,
                *order_columns,
                self.time_tokenizer.transform(pl.col("time_diffs")).alias("time_diffs"),
                *[
                    s.transform(pl.col(n)).alias(n)
                    for n, s in self.stringifiers.items()
                ],
            )
            .with_columns(
                pl.concat_list("time_diffs", *[pl.col(n) for n in self.stringifiers])
                .list.drop_nulls()
                .alias("feature_list")
            )
            .select(*keys, *targets, *order_columns, "feature_list")
            .explode("feature_list")
            .with_columns(
                pl.col("feature_list")
                .replace_strict(
                    self.complete_vocab, default=self.complete_vocab["<UNK>"]
                )
                .cast(pl.Int64)
                .alias("processed_list"),
            )
        )

        df = (
            df.sort(*order_columns)
            .group_by(*keys, *targets)  # Within-group order is always kept
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

    def __len__(self) -> int:
        """Obvious"""
        return self.df.height

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        """Return a single instance of the training dataset, as a dictionary."""

        row = self.df.row(idx, named=True)
        x = torch.nn.functional.pad(
            torch.tensor(row["processed_list"]),
            (0, self.max_length - row["sequence_length"]),
            value=self.complete_vocab["<PAD>"],
        )
        targets = (
            {t: torch.tensor(row[t]) for t in self.targets}
            if len(self.targets) > 0
            else {}
        )

        return (
            {k: row[k] for k in self.keys}
            | targets
            | {"x": x, "sequence_length": torch.tensor(row["sequence_length"])}
        )

    @classmethod
    def from_saved(cls, path, *args, **kwargs):
        raise NotImplementedError("TKTK load from saved")

    def get_state(self):
        return {
            "stringifiers": self.stringifiers,
            "time_tokenizer": self.time_tokenizer,
            "complete_vocab": self.complete_vocab,
            "max_seq_len": self.max_length,
        }

    def data_stats(self):

        n_tokens = self.df["sequence_length"].sum()
        n_possible_tokens = len(self) * self.max_length

        return {
            "n_sequences": len(self),
            "n_tokens": n_tokens,
            "average_fill": n_tokens / n_possible_tokens,
        }


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

    # Test new features

    def extra(df, keys):

        return df.with_columns(
            pl.col("events")
            .replace_strict(EVENT_IS_OUT, return_dtype=pl.Float32)
            .max()
            .over(partition_by=keys)
            .alias("is_out")
        )

    df_2 = TrainingDataset(
        path=config["prepared_data_path"],
        **config["dataset_params"],
        state_dict=df.get_state(),
        extra_processing=extra,
        targets=["is_out"],
    )

    print(len(df_2))
    print(df_2[500])

"""Tools for producing discrete sequences."""

from itertools import chain
from typing import Self, Any

import polars as pl
import numpy as np


class NumericState:

    DEFAULT_BUCKETS = 10

    def __init__(self, state: Any):
        self.state = state

    @classmethod
    def from_data(
        cls,
        column: pl.Expr,
        group: str,
        explicit_missing: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> Self:

        kwargs = {} if kwargs is None else kwargs
        n_buckets = kwargs.get("n_buckets", cls.DEFAULT_BUCKETS)
        print(f"{n_buckets=}")

        # We don't want the first or last one.
        cuts = np.linspace(0, 1, num=n_buckets + 2)[1:-1]
        quantiles = [column.quantile(q) for q in cuts]
        labels = (
            [f"(-inf)-{quantiles[0]}"]
            + [f"{ql}-{qh}" for ql, qh in zip(quantiles[:-1], quantiles[1:])]
            + [f"{quantiles[-1]}-(inf)"]
        )
        return cls(
            {
                "quantiles": quantiles,
                "values": labels,
                "group": group,
            }
        )

    def transform(self, x: pl.Expr) -> pl.Expr:

        return pl.concat_str(
            pl.lit(f"{self.state['group']} = "),
            x.cut(self.state["quantiles"], labels=self.state["values"]),
        )

    @property
    def vocab(self):
        return [f"{self.state['group']} = {label}" for label in self.state["values"]]


class CategoricalState:

    def __init__(self, state: Any):
        self.state = state

    @classmethod
    def from_data(
        cls,
        column: pl.Expr,
        group: str,
        explicit_missing: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> Self:

        unique = column.unique().to_list()

        return cls(
            {
                "values": unique,
                "group": group,
            }
        )

    def transform(self, x) -> pl.Expr:

        return pl.concat_str(pl.lit(f"{self.state['group']} = "), x)

    @property
    def vocab(self):
        return [f"{self.state['group']} = {label}" for label in self.state["values"]]


class Stringifier:
    """Turns features into strings."""

    ALLOWED_TYPES = {
        "categorical": CategoricalState,
        "numeric": NumericState,
    }

    def __init__(self, state):
        """Initialize directly from a state dict."""
        self._state = state

    @classmethod
    def from_data(
        cls,
        column: pl.Expr,
        data_type: str,
        group: str,
        explicit_missing: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> Self:
        """Make a stringifier from data."""

        keyword_args = {} if kwargs is None else kwargs

        assert data_type in cls.ALLOWED_TYPES

        state_dict = cls.ALLOWED_TYPES[data_type].from_data(
            column,
            group,
            explicit_missing,
            keyword_args,
        )

        return cls(state_dict)

    def transform(self, column: pl.Expr):
        return self._state.transform(column)

    @property
    def vocab(self):
        return self._state.vocab

    @property
    def state(self):
        return self._state.state


class TimeTokenizer:
    """Turns time gaps into tokens.

    Notably, gaps of 0 just disappear."""

    DEFAULT_BUCKETS = 32

    def __init__(self, state):
        self.state = state

    @classmethod
    def from_data(cls, column: pl.Expr, discretize: bool = False) -> Self:
        gaps = column.unique().drop_nulls().to_list()
        return cls({"values": gaps})

    def transform(self, column: pl.Expr):
        return (
            pl.when(column > 0)
            .then(pl.concat_str(pl.lit("GAP = "), column))
            .otherwise(pl.lit(None))
        )

    @property
    def vocab(self):
        return [f"GAP = {v}" for v in self.state["values"]]


def make_vocabulary(
    stringifiers: list[Stringifier],
    time_tokenizer: TimeTokenizer,
) -> dict[str, int]:

    unique_values = set(
        list(chain.from_iterable(s.vocab for s in stringifiers)) + time_tokenizer.vocab
    )

    return {item: i for i, item in enumerate(unique_values)}


if __name__ == "__main__":

    import numpy as np

    key = np.repeat(np.arange(10), 10)
    x = np.arange(100)
    y = np.tile(np.arange(5), 20)
    times = np.repeat(np.arange(50), 2)
    df = pl.DataFrame({"key": key, "x": x, "y": y, "times": times})

    # numeric
    xs = Stringifier.from_data(df["x"], "numeric", "X")
    df = df.with_columns()
    print(xs.state)
    print(xs.vocab)

    # Categorical
    ys = Stringifier.from_data(df["y"], "categorical", "Y")
    df = df.with_columns(ys.transform(pl.col("y")).alias("yt"))
    print(ys.state)
    print(ys.vocab)

    # Time
    df = df.with_columns(
        (pl.col("times") - pl.col("times").shift(1))
        .over(partition_by=pl.col("key"), order_by=pl.col("times"))
        .alias("time_diffs")
    )
    ts = TimeTokenizer.from_data(df["time_diffs"])
    print(ts.state)
    print(ts.vocab)

    # Transformation
    df = df.with_columns(
        xs.transform(pl.col("x")).alias("xt"),
        ys.transform(pl.col("y")).alias("yt"),
        ts.transform(pl.col("time_diffs")).alias("tt"),
    ).with_columns(
        pl.concat_list(pl.col("tt"), pl.col("xt"), pl.col("yt"))
        .list.drop_nulls()
        .alias("feature_list"),
    )

    print(df)

    # Vocabulary
    combined_vocab = make_vocabulary(
        stringifiers=[xs, ys],
        time_tokenizer=ts,
    )
    print(len(combined_vocab))

    simple_df = (
        df.select(pl.col("key"), pl.col("times"), pl.col("feature_list"))
        .explode("feature_list")
        .with_columns(
            pl.col("feature_list").replace(combined_vocab).alias("processed_list")
        )
    )

    print(simple_df)

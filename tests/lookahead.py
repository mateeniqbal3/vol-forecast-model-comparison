"""
Perturbation check for look-ahead bias.

A transform is causal if its output at date ``t`` does not change when every
input row strictly after ``t`` is altered. Rather than inspecting code for
future indexing, this perturbs the future, recomputes, and compares the past.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

Transform = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]


def leaking_cut_dates(
    transform: Transform,
    data: pd.DataFrame,
    cut_positions: Sequence[int],
    seed: int = 0,
) -> list[pd.Timestamp]:
    """
    Return every cut date at which scaling all numeric inputs strictly after
    the cut changed some output dated at or before the cut.
    """
    rng = np.random.default_rng(seed)
    numeric = data.select_dtypes("number").columns
    baseline = transform(data)
    leaks = []
    for pos in cut_positions:
        cut = data.index[pos]
        future = data.index > cut
        perturbed = data.copy()
        factors = rng.uniform(0.5, 1.5, size=(int(future.sum()), len(numeric)))
        perturbed.loc[future, numeric] = perturbed.loc[future, numeric].to_numpy() * factors

        recomputed = transform(perturbed)
        past_before = baseline.loc[baseline.index <= cut].to_numpy(dtype=float)
        past_after = recomputed.loc[recomputed.index <= cut].to_numpy(dtype=float)
        if past_before.shape != past_after.shape or not np.allclose(
            past_before, past_after, rtol=1e-12, atol=0.0, equal_nan=True
        ):
            leaks.append(cut)
    return leaks


def assert_no_lookahead(transform: Transform, data: pd.DataFrame, n_cuts: int = 15) -> None:
    cut_positions = np.linspace(10, len(data) - 10, n_cuts).astype(int)
    leaks = leaking_cut_dates(transform, data, cut_positions)
    assert not leaks, f"output at or before these dates depends on later data: {leaks[:5]}"

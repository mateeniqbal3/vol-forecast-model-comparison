"""
Tests for the forward realized-variance target, including the look-ahead
checks required by PROJECT.md section 7:

1. The target at t uses only returns strictly after t (t+1 .. t+h).
2. No feature construction uses data after t. Every feature builder is run
   through the same perturbation harness; the target itself is included as a
   positive control that the harness can detect a forward window.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import add_log_returns
from src.target import HORIZON, forward_realized_variance
from tests.lookahead import assert_no_lookahead, leaking_cut_dates


def _returns(values):
    return pd.Series(values, index=pd.bdate_range("2021-01-01", periods=len(values)))


def test_target_matches_hand_computation():
    r = _returns([0.01, -0.02, 0.03, 0.00, -0.01, 0.02, 0.01])
    target = forward_realized_variance(r, horizon=2)
    # Row 0 -> r1^2 + r2^2; row 4 -> r5^2 + r6^2; last two rows incomplete.
    assert target.iloc[0] == pytest.approx(0.02**2 + 0.03**2)
    assert target.iloc[4] == pytest.approx(0.02**2 + 0.01**2)
    assert target.iloc[-2:].isna().all()
    assert target.iloc[:-2].notna().all()


def test_horizon_one_is_next_squared_return():
    r = _returns(np.linspace(-0.02, 0.02, 20))
    target = forward_realized_variance(r, horizon=1)
    np.testing.assert_allclose(target.iloc[:-1].to_numpy(), r.iloc[1:].to_numpy() ** 2)


def test_default_horizon_leaves_exactly_h_trailing_nans(synthetic_prices):
    r = add_log_returns(synthetic_prices)["log_return"]
    target = forward_realized_variance(r)
    assert target.isna().sum() == HORIZON
    assert target.iloc[-HORIZON:].isna().all()
    assert target.name == f"rv_fwd_{HORIZON}d"


def test_missing_return_inside_window_gives_nan_not_partial_sum():
    r = _returns([0.01] * 10)
    r.iloc[5] = np.nan
    target = forward_realized_variance(r, horizon=3)
    # Rows 2, 3, 4 have r_5 inside their forward window.
    assert target.iloc[2:5].isna().all()
    assert target.iloc[1] == pytest.approx(3 * 0.01**2)


@pytest.mark.parametrize("horizon", [1, 3, HORIZON, 10])
def test_target_uses_only_returns_strictly_after_t(horizon):
    """Changing r_s must change exactly the targets at t = s-h .. s-1 and nothing else."""
    rng = np.random.default_rng(1)
    r = _returns(rng.normal(0, 0.01, 60))
    base = forward_realized_variance(r, horizon)

    for s in [15, 30, 45]:
        bumped = r.copy()
        bumped.iloc[s] += 0.05
        changed = ~np.isclose(
            forward_realized_variance(bumped, horizon).to_numpy(), base.to_numpy(), equal_nan=True
        )
        assert np.flatnonzero(changed).tolist() == list(range(s - horizon, s))


def test_target_is_unaffected_by_the_same_day_return():
    """r_t is known at t, so it must not be part of the target at t (strictly after)."""
    r = _returns(np.full(30, 0.01))
    bumped = r.copy()
    bumped.iloc[20] = 0.5
    base, new = forward_realized_variance(r), forward_realized_variance(bumped)
    assert new.iloc[20] == base.iloc[20]
    assert new.iloc[19] != base.iloc[19]


def test_price_change_after_t_plus_h_does_not_move_target_at_t(synthetic_prices):
    """Price-level version: P_{t+h+1} only enters returns from t+h+1 on."""
    t, h = 100, HORIZON

    def target_from_prices(prices):
        return forward_realized_variance(add_log_returns(prices)["log_return"], h)

    base = target_from_prices(synthetic_prices)
    bumped_prices = synthetic_prices.copy()
    bumped_prices.iloc[t + h + 1:, bumped_prices.columns.get_loc("close")] *= 1.3
    bumped = target_from_prices(bumped_prices)

    date_t = synthetic_prices.index[t]
    pd.testing.assert_series_equal(base.loc[:date_t], bumped.loc[:date_t])
    # Positive control: the next day's target does see the change.
    assert bumped.loc[synthetic_prices.index[t + 1]] != base.loc[synthetic_prices.index[t + 1]]


def test_rejects_bad_inputs():
    r = _returns([0.01] * 10)
    with pytest.raises(ValueError, match="horizon"):
        forward_realized_variance(r, horizon=0)
    with pytest.raises(ValueError, match="sorted"):
        forward_realized_variance(r.iloc[::-1])


# --- Features must never see the target's forward window --------------------

# Every transform that produces model inputs must be listed here. Phase 2/3
# feature builders are added as they are written.
FEATURE_BUILDERS = {
    "log_returns": lambda prices: add_log_returns(prices)["log_return"],
}


@pytest.mark.parametrize("name", sorted(FEATURE_BUILDERS))
def test_feature_builders_have_no_lookahead(name, synthetic_prices):
    assert_no_lookahead(FEATURE_BUILDERS[name], synthetic_prices)


def test_lookahead_harness_detects_the_forward_target(synthetic_prices):
    """Positive control: the harness must flag the target, or it proves nothing about features."""

    def target_from_prices(prices):
        return forward_realized_variance(add_log_returns(prices)["log_return"])

    cuts = np.linspace(10, len(synthetic_prices) - 10, 15).astype(int)
    assert len(leaking_cut_dates(target_from_prices, synthetic_prices, cuts)) == len(cuts)

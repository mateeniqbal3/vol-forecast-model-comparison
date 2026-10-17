"""Generate the figures used in paper/writeup.md and the README.

Reads only the committed result file docs/phase5_comparison.json. It does
not download data, refit any model, or change any recorded result.

Usage (from the repository root):
    python paper/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

res = json.loads((ROOT / "docs" / "phase5_comparison.json").read_text())

GARCH_C, LGBM_C, EWMA_C, INK = "#4C72B0", "#C44E52", "#8C8C8C", "#333333"
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def scheme_effect(ax):
    """QLIKE improvement over EWMA on the 1,321 common dates, by scheme."""
    t = res["table"]
    base = {s: t["EWMA"][s]["qlike"] for s in ("naive_same_dates", "walk_forward_same_dates")}
    models = [("GARCH(1,1)", GARCH_C), ("LightGBM", LGBM_C)]
    schemes = [
        ("naive_same_dates", "Naive random split"),
        ("walk_forward_same_dates", "Walk-forward"),
    ]
    width = 0.36
    for j, (key, label) in enumerate(schemes):
        for i, (m, color) in enumerate(models):
            v = t[m][key]["qlike"] - base[key]
            x = i + (j - 0.5) * width
            ax.bar(
                x,
                v,
                width=width - 0.03,
                color=color,
                alpha=1.0 if j else 0.45,
                edgecolor=color,
                hatch="" if j else "///",
                linewidth=1,
            )
            ax.text(x, v - 0.003, f"{v:+.3f}", ha="center", va="top", fontsize=8, color=INK)
            ax.text(
                x,
                0.004,
                "naive" if j == 0 else "walk-\nforward",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([0, 1], ["GARCH(1,1)", "LightGBM"])
    ax.set_ylabel("Mean QLIKE minus EWMA (lower is better)")
    ax.set_ylim(-0.145, 0.03)
    ax.set_title(
        "Same 1,321 dates: the naive split flatters\nLightGBM eight times as much as GARCH"
    )
    ax.grid(axis="x", visible=False)


def ceiling(ax):
    """Peak realized variance vs each model's peak forecast, per walk-forward fold."""
    folds = res["walk_forward_extrapolation"]["folds"]
    yrs = [f["year"] for f in folds]
    ax.step(
        yrs,
        [f["max_training_rv"] for f in folds],
        where="mid",
        color=INK,
        lw=1,
        ls="--",
        label="Largest target in the training set",
    )
    ax.plot(
        yrs,
        [f["max_realized_rv"] for f in folds],
        "o",
        color=INK,
        ms=4,
        mfc="white",
        label="Peak realized 5-day variance",
    )
    ax.plot(
        yrs,
        [f["max_forecast_GARCH(1,1)"] for f in folds],
        "-",
        color=GARCH_C,
        lw=1.8,
        label="Peak GARCH(1,1) forecast",
    )
    ax.plot(
        yrs,
        [f["max_forecast_LightGBM"] for f in folds],
        "-",
        color=LGBM_C,
        lw=1.8,
        label="Peak LightGBM forecast",
    )
    ax.set_yscale("log")
    ax.set_ylabel("5-day variance (log scale)")
    ax.set_xlim(1999.3, 2025.7)
    f08 = next(f for f in folds if f["year"] == 2008)
    ax.annotate(
        f"2008: realized {f08['max_realized_rv']:.3f}\nLightGBM peak {f08['max_forecast_LightGBM']:.4f}",
        (2008, f08["max_forecast_LightGBM"]),
        xytext=(2009.4, 0.00022),
        fontsize=8,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.8},
    )
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.set_ylim(1.5e-4, 0.08)
    ax.set_title(
        "Walk-forward: a tree ensemble cannot forecast\nbeyond the range it was trained on"
    )


def yearly_contribution(ax):
    """Each year's contribution to the full-sample LightGBM-minus-GARCH QLIKE difference."""
    q = res["walk_forward_qlike_diff_lgbm_minus_garch"]
    yrs = sorted(q["by_year"], key=int)
    vals = [q["by_year"][y] for y in yrs]
    hot = {str(y) for y in q["extrapolation_years"]}
    colors = [
        LGBM_C if y in hot else ("#9DB3D6" if v < 0 else "#E3A5A7") for y, v in zip(yrs, vals)
    ]
    ax.bar([int(y) for y in yrs], vals, color=colors, width=0.75)
    ax.axhline(0, color="black", lw=0.8)
    for y in hot:
        v = q["by_year"][y]
        ax.text(
            int(y), v + 0.0006, f"{y}\n{v:+.4f}", ha="center", va="bottom", fontsize=8, color=INK
        )
    ax.text(
        2014.5,
        0.021,
        f"Sum over all years: {q['mean']:+.4f}\n(negative favours LightGBM; DM p = 0.47)",
        ha="center",
        fontsize=8.5,
        color=INK,
    )
    ax.set_ylabel("Contribution to mean QLIKE difference\n(LightGBM minus GARCH)")
    ax.set_ylim(-0.008, 0.032)
    ax.set_xlim(1999.3, 2025.7)
    ax.set_title("2008 alone costs LightGBM more than twice its net advantage")
    ax.grid(axis="x", visible=False)


figs = [
    ("fig1_scheme_effect.png", scheme_effect, (5.2, 4.0)),
    ("fig2_forecast_ceiling.png", ceiling, (6.6, 4.0)),
    ("fig3_yearly_contribution.png", yearly_contribution, (6.6, 3.6)),
]
for name, fn, size in figs:
    fig, ax = plt.subplots(figsize=size)
    fn(ax)
    fig.savefig(OUT / name)
    plt.close(fig)

# README hero: figures 1 and 2 side by side.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1, 1.35]})
scheme_effect(axes[0])
ceiling(axes[1])
fig.suptitle(
    "SPY 5-day volatility, 2000-2025: LightGBM's apparent edge over GARCH(1,1) "
    "disappears under walk-forward validation",
    fontweight="bold",
)
fig.tight_layout()
fig.savefig(OUT / "readme_hero.png")
plt.close(fig)

print(f"Wrote figures to {OUT.relative_to(ROOT)}")

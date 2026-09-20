"""
STEP 80 — REPRODUCIBLE MANUSCRIPT FIGURE GENERATION

Purpose
-------
Generate publication-quality figures directly from the saved
experimental outputs.

Figures
-------
Figure 2:
    Exhaustive 495 four-sensor configuration performance.

Figure 3:
    Ten-seed CA-RDQN performance versus zero-change reference.

Figure 4:
    Paired CA-RDQN versus CA-DQN recurrence ablation.

Figure 5:
    CA-RDQN policy diversity across ten independent seeds.

Outputs
-------
data/processed/manuscript_figures/

    Figure_2_495_configuration_performance.pdf
    Figure_2_495_configuration_performance.png

    Figure_3_CA_RDQN_vs_zero_change.pdf
    Figure_3_CA_RDQN_vs_zero_change.png

    Figure_4_CA_RDQN_vs_CA_DQN_ablation.pdf
    Figure_4_CA_RDQN_vs_CA_DQN_ablation.png

    Figure_5_CA_RDQN_policy_diversity.pdf
    Figure_5_CA_RDQN_policy_diversity.png

IMPORTANT
---------
No model training is performed.
No experimental values are manually entered into the plots.
The figures are regenerated from saved experimental outputs.
"""

from pathlib import Path
import json
import math
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RDQN_DIR = (
    ROOT
    / "data"
    / "processed"
    / "final_ca_rdqn"
)

DQN_DIR = (
    ROOT
    / "data"
    / "processed"
    / "final_ca_dqn_ablation"
)

ANALYSIS_DIR = (
    ROOT
    / "data"
    / "processed"
    / "ca_rdqn_vs_ca_dqn_analysis"
)

FIGURE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "manuscript_figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FINAL RANDOM SEEDS
# ============================================================

SEEDS = [
    11,
    22,
    33,
    44,
    55,
    66,
    77,
    88,
    99,
    111,
]


# ============================================================
# PUBLICATION SETTINGS
# ============================================================

DPI = 600

FIGURE_WIDTH = 7.2

FIGURE_HEIGHT = 4.8


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def load_json(path):
    """Load JSON file."""

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_figure(fig, stem):
    """
    Save both vector PDF and high-resolution PNG.
    """

    pdf_path = (
        FIGURE_DIR
        / f"{stem}.pdf"
    )

    png_path = (
        FIGURE_DIR
        / f"{stem}.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"[SAVED] {pdf_path}"
    )

    print(
        f"[SAVED] {png_path}"
    )


def load_seed_result(
    directory,
    prefix,
    seed,
):
    """
    Load one seed-level result file.
    """

    path = (
        directory
        / f"{prefix}_{seed}_results.json"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing result file:\n{path}"
        )

    return load_json(path)


# ============================================================
# LOAD ALL FINAL RESULTS
# ============================================================

print("=" * 78)

print(
    "STEP 80 — REPRODUCIBLE MANUSCRIPT FIGURE GENERATION"
)

print("=" * 78)

rdqn_results = {}

dqn_results = {}


for seed in SEEDS:

    rdqn_results[seed] = load_seed_result(
        RDQN_DIR,
        "ca_rdqn_seed",
        seed,
    )

    dqn_results[seed] = load_seed_result(
        DQN_DIR,
        "ca_dqn_ablation_seed",
        seed,
    )


print(
    "[PASS] Loaded all ten CA-RDQN result files."
)

print(
    "[PASS] Loaded all ten CA-DQN result files."
)


# ============================================================
# EXTRACT CA-RDQN RESULTS
# ============================================================

rdqn_nh4 = np.array(
    [
        rdqn_results[s]["nh4_rmse"]
        for s in SEEDS
    ],
    dtype=float,
)

rdqn_sno = np.array(
    [
        rdqn_results[s]["sno_rmse"]
        for s in SEEDS
    ],
    dtype=float,
)

rdqn_reward = np.array(
    [
        rdqn_results[s]["mean_evaluation_reward"]
        for s in SEEDS
    ],
    dtype=float,
)

rdqn_unique = np.array(
    [
        rdqn_results[s]["unique_evaluation_actions"]
        for s in SEEDS
    ],
    dtype=float,
)


# ============================================================
# EXTRACT CA-DQN RESULTS
# ============================================================

dqn_nh4 = np.array(
    [
        dqn_results[s]["nh4_rmse"]
        for s in SEEDS
    ],
    dtype=float,
)

dqn_sno = np.array(
    [
        dqn_results[s]["sno_rmse"]
        for s in SEEDS
    ],
    dtype=float,
)

dqn_reward = np.array(
    [
        dqn_results[s]["mean_evaluation_reward"]
        for s in SEEDS
    ],
    dtype=float,
)

dqn_unique = np.array(
    [
        dqn_results[s]["unique_evaluation_actions"]
        for s in SEEDS
    ],
    dtype=float,
)


# ============================================================
# VERIFY BASELINES
# ============================================================

nh4_baselines = np.array(
    [
        rdqn_results[s][
            "nh4_zero_change_baseline"
        ]
        for s in SEEDS
    ],
    dtype=float,
)

sno_baselines = np.array(
    [
        rdqn_results[s][
            "sno_zero_change_baseline"
        ]
        for s in SEEDS
    ],
    dtype=float,
)


if not np.allclose(
    nh4_baselines,
    nh4_baselines[0],
):
    raise ValueError(
        "NH4 zero-change baseline is not identical "
        "across seeds."
    )


if not np.allclose(
    sno_baselines,
    sno_baselines[0],
):
    raise ValueError(
        "SNO zero-change baseline is not identical "
        "across seeds."
    )


NH4_BASELINE = nh4_baselines[0]

SNO_BASELINE = sno_baselines[0]


print(
    "[PASS] Zero-change baselines verified."
)


# ============================================================
# FIGURE 3
# CA-RDQN VS ZERO-CHANGE REFERENCE
# ============================================================

print()
print(
    "Generating Figure 3..."
)


fig, ax = plt.subplots(
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT,
    )
)


x = np.arange(
    len(SEEDS)
)


offset = 0.18


ax.scatter(
    x - offset,
    np.full(
        len(SEEDS),
        NH4_BASELINE,
    ),
    marker="o",
    s=42,
    label="ΔNH4 zero-change",
)


ax.scatter(
    x + offset,
    rdqn_nh4,
    marker="o",
    s=42,
    label="ΔNH4 CA-RDQN",
)


ax.plot(
    x + offset,
    rdqn_nh4,
    linewidth=1.0,
)


ax.axhline(
    NH4_BASELINE,
    linestyle="--",
    linewidth=1.0,
)


ax.set_xticks(x)

ax.set_xticklabels(
    [str(s) for s in SEEDS]
)

ax.set_xlabel(
    "Random seed"
)

ax.set_ylabel(
    "ΔNH4 RMSE"
)

ax.set_title(
    "CA-RDQN performance across independent seeds"
)

ax.legend(
    frameon=False,
)

ax.grid(
    axis="y",
    alpha=0.25,
)

save_figure(
    fig,
    "Figure_3_CA_RDQN_vs_zero_change",
)


# ============================================================
# FIGURE 4
# CA-RDQN VS CA-DQN ABLATION
# ============================================================

print()
print(
    "Generating Figure 4..."
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        7.2,
        3.8,
    ),
)


# ------------------------------------------------------------
# ΔNH4
# ------------------------------------------------------------

ax = axes[0]


for i in range(
    len(SEEDS)
):

    ax.plot(
        [0, 1],
        [
            rdqn_nh4[i],
            dqn_nh4[i],
        ],
        marker="o",
        linewidth=0.9,
    )


ax.set_xticks(
    [0, 1]
)

ax.set_xticklabels(
    [
        "CA-RDQN",
        "CA-DQN",
    ]
)

ax.set_ylabel(
    "ΔNH4 RMSE"
)

ax.set_title(
    "ΔNH4"
)

ax.grid(
    axis="y",
    alpha=0.25,
)


# ------------------------------------------------------------
# ΔSNO
# ------------------------------------------------------------

ax = axes[1]


for i in range(
    len(SEEDS)
):

    ax.plot(
        [0, 1],
        [
            rdqn_sno[i],
            dqn_sno[i],
        ],
        marker="o",
        linewidth=0.9,
    )


ax.set_xticks(
    [0, 1]
)

ax.set_xticklabels(
    [
        "CA-RDQN",
        "CA-DQN",
    ]
)

ax.set_ylabel(
    "ΔSNO RMSE"
)

ax.set_title(
    "ΔSNO"
)

ax.grid(
    axis="y",
    alpha=0.25,
)


fig.suptitle(
    "Paired ten-seed recurrence ablation",
)

fig.tight_layout()


save_figure(
    fig,
    "Figure_4_CA_RDQN_vs_CA_DQN_ablation",
)


# ============================================================
# FIGURE 5
# POLICY DIVERSITY
# ============================================================

print()
print(
    "Generating Figure 5..."
)


fig, ax = plt.subplots(
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT,
    ),
)


x = np.arange(
    len(SEEDS)
)

width = 0.36


ax.bar(
    x - width / 2,
    rdqn_unique,
    width,
    label="CA-RDQN",
)


ax.bar(
    x + width / 2,
    dqn_unique,
    width,
    label="CA-DQN",
)


ax.set_xticks(x)

ax.set_xticklabels(
    [str(s) for s in SEEDS]
)

ax.set_xlabel(
    "Random seed"
)

ax.set_ylabel(
    "Unique evaluation actions"
)

ax.set_title(
    "Policy diversity across independent seeds"
)

ax.legend(
    frameon=False,
)

ax.grid(
    axis="y",
    alpha=0.25,
)

save_figure(
    fig,
    "Figure_5_CA_RDQN_policy_diversity",
)


## ============================================================
# FIGURE 2
# 495-CONFIGURATION PERFORMANCE
# ============================================================

print()
print(
    "Generating Figure 2..."
)

BENCHMARK_FILE = (
    ROOT
    / "data"
    / "processed"
    / "benchmark_4of12_results.csv"
)

if not BENCHMARK_FILE.exists():

    raise FileNotFoundError(
        f"Required exhaustive benchmark file not found: "
        f"{BENCHMARK_FILE}"
    )

benchmark = pd.read_csv(
    BENCHMARK_FILE
)

# ------------------------------------------------------------
# Validate the saved benchmark
# ------------------------------------------------------------

if len(benchmark) != 495:

    raise ValueError(
        "Expected exactly 495 four-sensor configurations, "
        f"but found {len(benchmark)}."
    )

required_columns = [
    "action_index",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "nh4_rmse",
    "sno_rmse",
]

missing_columns = [
    column
    for column in required_columns
    if column not in benchmark.columns
]

if missing_columns:

    raise ValueError(
        "Missing required benchmark columns: "
        + ", ".join(missing_columns)
    )

if not np.all(
    np.isfinite(
        benchmark[
            [
                "nh4_rmse",
                "sno_rmse",
            ]
        ].to_numpy()
    )
):

    raise ValueError(
        "Non-finite RMSE values detected in "
        "the exhaustive benchmark."
    )

print(
    f"[PASS] Loaded exhaustive benchmark: "
    f"{BENCHMARK_FILE}"
)

print(
    "[PASS] Verified 495 four-sensor configurations."
)


# ------------------------------------------------------------
# Figure 2 — RMSE distributions
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        FIGURE_WIDTH * 1.35,
        FIGURE_HEIGHT,
    )
)

nh4_values = benchmark["nh4_rmse"].to_numpy()
sno_values = benchmark["sno_rmse"].to_numpy()


# ------------------------------------------------------------
# NH4-N
# ------------------------------------------------------------

axes[0].hist(
    nh4_values,
    bins=25,
    edgecolor="black",
    linewidth=0.7,
)

axes[0].axvline(
    nh4_values.mean(),
    linestyle="--",
    linewidth=1.5,
    label=f"Mean = {nh4_values.mean():.3f}",
)

axes[0].axvline(
    nh4_values.min(),
    linestyle=":",
    linewidth=1.5,
    label=f"Best = {nh4_values.min():.3f}",
)

axes[0].set_xlabel(
    r"$\Delta$NH$_4$-N RMSE"
)

axes[0].set_ylabel(
    "Number of configurations"
)

axes[0].set_title(
    "Four-sensor configuration performance: NH$_4$-N"
)

axes[0].legend(
    frameon=False,
)


# ------------------------------------------------------------
# SNO
# ------------------------------------------------------------

axes[1].hist(
    sno_values,
    bins=25,
    edgecolor="black",
    linewidth=0.7,
)

axes[1].axvline(
    sno_values.mean(),
    linestyle="--",
    linewidth=1.5,
    label=f"Mean = {sno_values.mean():.3f}",
)

axes[1].axvline(
    sno_values.min(),
    linestyle=":",
    linewidth=1.5,
    label=f"Best = {sno_values.min():.3f}",
)

axes[1].set_xlabel(
    r"$\Delta$SNO RMSE"
)

axes[1].set_ylabel(
    "Number of configurations"
)

axes[1].set_title(
    "Four-sensor configuration performance: SNO"
)

axes[1].legend(
    frameon=False,
)


# ------------------------------------------------------------
# Common formatting
# ------------------------------------------------------------

for ax in axes:

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


fig.tight_layout()

save_figure(
    fig,
    "Figure_2_495_configuration_performance",
)

print(
    "[PASS] Figure 2 generated from the "
    "saved 495-configuration benchmark."
)

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 78)

print(
    "STEP 80 FIGURE GENERATION SUMMARY"
)

print("=" * 78)

print()
print(
    "Figure directory:"
)

print(
    FIGURE_DIR
)

print()
print(
    "[PASS] Figure 2 generated."
)

print(
    "[PASS] Figure 3 generated."
)

print(
    "[PASS] Figure 4 generated."
)

print(
    "[PASS] Figure 5 generated."
)

print()
print(
    "[PASS] No model training was performed."
)

print(
    "[PASS] Figures use saved experimental results."
)

print(
    "[PASS] PNG outputs use 600 dpi."
)

print(
    "[PASS] PDF outputs are vector-based."
)

print()
print(
    "STEP 80 COMPLETE."
)
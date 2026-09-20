"""
Figure 6 — Temporal variation in the best-performing four-sensor configuration.

Source:
data/processed/temporal_sensor_oracle/temporal_oracle_results.json

The figure uses only the results produced by temporal_sensor_oracle.py.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "temporal_sensor_oracle"
    / "temporal_oracle_results.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "manuscript_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Load results
# ------------------------------------------------------------

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
) as f:
    results = json.load(f)


days = np.array(
    [item["day"] for item in results],
    dtype=int,
)

nh4_rmse = np.array(
    [item["nh4_rmse"] for item in results],
    dtype=float,
)

sno_rmse = np.array(
    [item["sno_rmse"] for item in results],
    dtype=float,
)


nh4_configs = [
    " + ".join(
        item["best_nh4_configuration"]
    )
    for item in results
]

sno_configs = [
    " + ".join(
        item["best_sno_configuration"]
    )
    for item in results
]


# ------------------------------------------------------------
# Publication settings
# ------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }
)


# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    sharex=True,
)


# ------------------------------------------------------------
# ΔNH4-N panel
# ------------------------------------------------------------

axes[0].plot(
    days,
    nh4_rmse,
    marker="o",
    linewidth=2.0,
    markersize=7,
    label="Best four-sensor configuration",
)

for x, y, config in zip(
    days,
    nh4_rmse,
    nh4_configs,
):

    axes[0].annotate(
        config,
        xy=(x, y),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        rotation=0,
    )


axes[0].set_ylabel(
    "RMSE of ΔNH₄-N"
)

axes[0].set_xticks(days)

axes[0].grid(
    axis="y",
    alpha=0.25,
)

axes[0].legend(
    loc="upper right",
    frameon=False,
)


# ------------------------------------------------------------
# ΔSNO panel
# ------------------------------------------------------------

axes[1].plot(
    days,
    sno_rmse,
    marker="o",
    linewidth=2.0,
    markersize=7,
    label="Best four-sensor configuration",
)

for x, y, config in zip(
    days,
    sno_rmse,
    sno_configs,
):

    axes[1].annotate(
        config,
        xy=(x, y),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        rotation=0,
    )


axes[1].set_ylabel(
    "RMSE of ΔSNO"
)

axes[1].set_xlabel(
    "Evaluation window (day)"
)

axes[1].set_xticks(days)

axes[1].grid(
    axis="y",
    alpha=0.25,
)

axes[1].legend(
    loc="upper right",
    frameon=False,
)


# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------

fig.tight_layout(
    h_pad=2.0
)


# ------------------------------------------------------------
# Save high-resolution outputs
# ------------------------------------------------------------

PNG_FILE = (
    OUTPUT_DIR
    / "Figure_6_temporal_oracle.png"
)

PDF_FILE = (
    OUTPUT_DIR
    / "Figure_6_temporal_oracle.pdf"
)


fig.savefig(
    PNG_FILE,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    PDF_FILE,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("=" * 70)
print("FIGURE 6 GENERATED")
print("=" * 70)
print()
print(f"Input:  {INPUT_FILE}")
print(f"PNG:    {PNG_FILE}")
print(f"PDF:    {PDF_FILE}")
print()
print("Five temporal evaluation windows plotted.")
print("ΔNH4-N and ΔSNO panels generated.")
print("PNG resolution: 600 dpi")
print("PDF: vector format")
print()
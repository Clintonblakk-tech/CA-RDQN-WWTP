"""
Figure 7 — CA-RDQN training reward across ten independent seeds.

Uses the final saved training histories only.
No training is rerun.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_ca_rdqn"
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


# ------------------------------------------------------------
# Load episode rewards
# ------------------------------------------------------------

reward_histories = []

for seed in SEEDS:

    history_file = (
        HISTORY_DIR
        / f"ca_rdqn_seed_{seed}_history.npz"
    )

    if not history_file.exists():
        raise FileNotFoundError(
            f"Missing history file: {history_file}"
        )

    data = np.load(
        history_file
    )

    rewards = np.asarray(
        data["episode_rewards"],
        dtype=float,
    )

    if rewards.shape != (24,):
        raise ValueError(
            f"Seed {seed}: expected 24 episode rewards, "
            f"found shape {rewards.shape}"
        )

    reward_histories.append(
        rewards
    )


reward_histories = np.vstack(
    reward_histories
)


# ------------------------------------------------------------
# Verify common experimental structure
# ------------------------------------------------------------

if reward_histories.shape != (10, 24):
    raise ValueError(
        "Unexpected reward-history matrix shape: "
        f"{reward_histories.shape}"
    )


mean_reward = np.mean(
    reward_histories,
    axis=0,
)

std_reward = np.std(
    reward_histories,
    axis=0,
    ddof=1,
)

episodes = np.arange(
    1,
    25,
)


# ------------------------------------------------------------
# Publication figure settings
# ------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    }
)


# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 6),
)


ax.plot(
    episodes,
    mean_reward,
    linewidth=2.5,
    marker="o",
    markersize=5,
    label="Mean across 10 seeds",
)


ax.fill_between(
    episodes,
    mean_reward - std_reward,
    mean_reward + std_reward,
    alpha=0.20,
    label="±1 SD",
)


ax.set_xlabel(
    "Training episode"
)

ax.set_ylabel(
    "Episode reward"
)

ax.set_xticks(
    episodes
)

ax.grid(
    axis="y",
    alpha=0.25,
)

ax.legend(
    loc="best",
    frameon=False,
)


fig.tight_layout()


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

PNG_FILE = (
    OUTPUT_DIR
    / "Figure_7_CA_RDQN_training_reward.png"
)

PDF_FILE = (
    OUTPUT_DIR
    / "Figure_7_CA_RDQN_training_reward.pdf"
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


# ------------------------------------------------------------
# Console verification
# ------------------------------------------------------------

print()
print("=" * 70)
print("FIGURE 7 GENERATED")
print("=" * 70)
print()

print(
    f"Seeds: {len(SEEDS)}"
)

print(
    "Episodes per seed: 24"
)

print(
    "Reward matrix:",
    reward_histories.shape,
)

print()

print(
    f"Mean reward, Episode 1: "
    f"{mean_reward[0]:.6f}"
)

print(
    f"Mean reward, Episode 24: "
    f"{mean_reward[-1]:.6f}"
)

print()

print(
    f"PNG: {PNG_FILE}"
)

print(
    f"PDF: {PDF_FILE}"
)

print()

print(
    "PNG resolution: 600 dpi"
)

print(
    "PDF: vector format"
)

print()

print(
    "No training was rerun."
)

print("=" * 70)
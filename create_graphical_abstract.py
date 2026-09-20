"""
Reproducible graphical abstract for:
Constraint-Aware Deep Q-Network for Dynamic Sensor Activation Scheduling
in Wastewater Treatment Plants: A Multi-Seed Robustness and Ablation Study

Requirements:
    Python 3.x
    matplotlib
    numpy

Run from VS Code PowerShell:
    python .\create_graphical_abstract.py

Outputs:
    outputs\graphical_abstract_energy_reports.png
    outputs\graphical_abstract_energy_reports.pdf
    outputs\graphical_abstract_energy_reports.svg
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, Circle, Rectangle, Polygon, FancyArrowPatch
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Publication figure settings
# ---------------------------------------------------------------------
FIG_W, FIG_H = 16, 9
DPI = 600

# Colour palette
NAVY = "#173F6D"
BLUE = "#3F8CC9"
GREEN = "#4C9A5F"
PURPLE = "#7353A8"
ORANGE = "#D88932"
RED = "#D9574E"
DARK = "#183247"
GREY = "#68727D"
WHITE = "#FFFFFF"

LIGHT_BLUE = "#EAF4FC"
LIGHT_GREEN = "#EEF8EF"
LIGHT_PURPLE = "#F2EEFA"
LIGHT_ORANGE = "#FBF0E4"

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=180)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")


# ---------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------
def rounded_box(x, y, w, h, face, edge="none", lw=1.0, radius=0.10):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.015,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw
    )
    ax.add_patch(patch)
    return patch


def arrow(x1, y1, x2, y2, colour=GREY, lw=2.2, size=12):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=size,
            linewidth=lw,
            color=colour
        )
    )


def panel(x, y, w, h, face, edge, title, number, number_colour):
    """Panel with a separate number badge and a wrapped, centred heading."""
    rounded_box(x, y, w, h, face, edge, 1.1, 0.15)

    ax.add_patch(
        Circle(
            (x + 0.38, y + h - 0.40),
            0.22,
            facecolor=number_colour,
            edgecolor="none",
            zorder=5
        )
    )

    ax.text(
        x + 0.38, y + h - 0.40, str(number),
        ha="center", va="center",
        fontsize=11.5, fontweight="bold",
        color=WHITE, zorder=6
    )

    ax.text(
        x + w / 2, y + h - 0.25, title,
        ha="center", va="top",
        fontsize=9.3, fontweight="bold",
        color=DARK, linespacing=1.05,
        zorder=5
    )


def neural_network(cx, cy, scale=1.0):
    layers = [
        [
            (cx - 0.55 * scale, cy + 0.35 * scale),
            (cx - 0.55 * scale, cy),
            (cx - 0.55 * scale, cy - 0.35 * scale),
        ],
        [
            (cx - 0.10 * scale, cy + 0.50 * scale),
            (cx - 0.10 * scale, cy + 0.17 * scale),
            (cx - 0.10 * scale, cy - 0.17 * scale),
            (cx - 0.10 * scale, cy - 0.50 * scale),
        ],
        [
            (cx + 0.35 * scale, cy + 0.35 * scale),
            (cx + 0.35 * scale, cy),
            (cx + 0.35 * scale, cy - 0.35 * scale),
        ],
    ]

    for i in range(len(layers) - 1):
        for p in layers[i]:
            for q in layers[i + 1]:
                ax.plot(
                    [p[0], q[0]], [p[1], q[1]],
                    color="#8A9097", linewidth=0.65, zorder=2
                )

    for layer in layers:
        for p in layer:
            ax.add_patch(
                Circle(
                    p, 0.065 * scale,
                    facecolor=WHITE,
                    edgecolor=DARK,
                    linewidth=0.8,
                    zorder=3
                )
            )


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
ax.text(
    8, 8.63,
    "Constraint-Aware Deep Q-Network for Dynamic Sensor Activation Scheduling",
    ha="center", va="center",
    fontsize=17.0, fontweight="bold", color=NAVY
)

ax.text(
    8, 8.29,
    "in Wastewater Treatment Plants: A Multi-Seed Robustness and Ablation Study",
    ha="center", va="center",
    fontsize=14.8, fontweight="bold", color=NAVY
)

ax.text(
    8, 7.97,
    "SMARTER MONITORING  |  EFFICIENT SENSOR USE  |  ROBUST EVALUATION",
    ha="center", va="center",
    fontsize=8.8, color=NAVY
)


# ---------------------------------------------------------------------
# Main four panels
# ---------------------------------------------------------------------
panel(
    0.12, 3.10, 3.75, 4.58,
    LIGHT_BLUE, "#C7DCEB",
    "BSM1 wastewater treatment\nprocess", 1, NAVY
)

panel(
    3.98, 3.10, 3.70, 4.58,
    LIGHT_GREEN, "#CBE1CE",
    "Candidate sensors and\nactivation constraint", 2, GREEN
)

panel(
    7.80, 3.10, 3.78, 4.58,
    LIGHT_PURPLE, "#D9CBEA",
    "Constraint-aware\ndeep Q-network", 3, PURPLE
)

panel(
    11.70, 3.10, 4.18, 4.58,
    LIGHT_ORANGE, "#EAD7BF",
    "Soft sensing\nand feedback", 4, ORANGE
)


# ---------------------------------------------------------------------
# PANEL 1 — BSM1
# ---------------------------------------------------------------------
ax.text(
    1.995, 6.78,
    "Conventional activated-sludge wastewater treatment plant",
    ha="center", fontsize=7.6, color=DARK
)

ax.text(0.30, 5.66, "Influent", fontsize=7.6, color=DARK)
arrow(0.70, 5.70, 1.00, 5.70, NAVY, 1.7, 9)

# Five biological reactor compartments
x0, y0 = 1.00, 4.76
for i in range(5):
    rx = x0 + i * 0.42

    ax.add_patch(
        Rectangle(
            (rx, y0), 0.42, 0.68,
            facecolor="#D5C4A2",
            edgecolor=DARK,
            linewidth=0.7
        )
    )

    rng = np.random.default_rng(100 + i)
    for bx, by in zip(
        rng.uniform(rx + 0.08, rx + 0.34, 7),
        rng.uniform(y0 + 0.10, y0 + 0.56, 7)
    ):
        ax.add_patch(
            Circle(
                (bx, by), 0.018,
                facecolor=WHITE,
                edgecolor="#AAB0B7",
                linewidth=0.3
            )
        )

    ax.text(
        rx + 0.21, y0 - 0.10,
        f"R{i + 1}",
        ha="center", va="top",
        fontsize=7.0, color=DARK
    )

# Reactor pipe
ax.plot(
    [1.03, 2.99], [5.58, 5.58],
    color=DARK, linewidth=0.9
)
for i in range(5):
    rx = x0 + i * 0.42 + 0.21
    ax.plot(
        [rx, rx], [5.58, 5.43],
        color=DARK, linewidth=0.7
    )

# Secondary settler
sx = 3.18
ax.add_patch(
    Polygon(
        [
            [sx, 5.35],
            [sx + 0.55, 5.35],
            [sx + 0.42, 4.76],
            [sx + 0.13, 4.76]
        ],
        closed=True,
        facecolor="#DDEAF3",
        edgecolor=DARK,
        linewidth=0.7
    )
)

arrow(3.00, 5.70, 3.38, 5.70, NAVY, 1.7, 9)
ax.text(3.08, 5.96, "Effluent", fontsize=7.6, color=DARK)

ax.text(
    2.00, 4.34,
    "Biological reactor (5 compartments)",
    ha="center", fontsize=7.4, color=DARK
)

ax.text(
    3.46, 4.34,
    "Secondary\nsettler",
    ha="center", fontsize=7.0, color=DARK
)

# BSM1 notes — deliberately short to prevent crowding
rounded_box(0.40, 3.48, 3.18, 0.72, "#DFF1E2")

notes = [
    "Nitrification and pre-denitrification",
    "Dynamic influent conditions",
    "Standard 14-day evaluation"
]

for j, text in enumerate(notes):
    ax.text(
        0.56, 4.02 - j * 0.22,
        "\u2713  " + text,
        fontsize=6.4,
        color=DARK,
        va="center"
    )


# ---------------------------------------------------------------------
# PANEL 2 — Candidate sensors
# ---------------------------------------------------------------------
sensor_labels = [
    (4.75, BLUE, "DO", "dissolved\noxygen"),
    (5.82, ORANGE, "NH₄-N", "ammonium\nnitrogen"),
    (6.89, GREEN, "SNO", "nitrate/nitrite\nnitrogen"),
]

for xpos, colour, label, subtitle in sensor_labels:
    ax.add_patch(
        Circle((xpos, 6.62), 0.20,
               facecolor=colour, edgecolor="none")
    )

    ax.text(
        xpos, 6.62, label,
        ha="center", va="center",
        fontsize=7.4, fontweight="bold",
        color=WHITE
    )

    ax.text(
        xpos, 6.30, subtitle,
        ha="center", va="top",
        fontsize=5.9, color=DARK
    )

# Sensor grid
sx0, sy = 4.50, 4.91
for i in range(5):
    rx = sx0 + i * 0.55

    ax.add_patch(
        Rectangle(
            (rx, sy), 0.55, 0.80,
            facecolor="#E2D5B7",
            edgecolor=DARK,
            linewidth=0.7
        )
    )

    ax.text(
        rx + 0.275, sy - 0.10,
        f"R{i + 1}",
        ha="center", va="top",
        fontsize=7.0, color=DARK
    )

    ax.add_patch(Circle((rx + 0.18, sy + 0.60), 0.062, fc=BLUE, ec="none"))
    ax.add_patch(Circle((rx + 0.38, sy + 0.40), 0.062, fc=ORANGE, ec="none"))
    ax.add_patch(Circle((rx + 0.28, sy + 0.20), 0.062, fc=GREEN, ec="none"))

ax.text(
    5.83, 4.39,
    "12 candidate sensors distributed across the biological reactor",
    ha="center", fontsize=6.9, color=DARK
)

rounded_box(4.38, 3.91, 2.92, 0.40, "#D8F0DA")
ax.text(
    5.83, 4.11,
    "Exactly four sensors active at each decision time",
    ha="center", va="center",
    fontsize=7.7, fontweight="bold", color=DARK
)

rounded_box(4.38, 3.40, 2.92, 0.40, "#F9E4CF")
ax.text(
    5.83, 3.60,
    "495 feasible four-sensor combinations",
    ha="center", va="center",
    fontsize=7.8, fontweight="bold", color=DARK
)

ax.text(
    5.83, 3.36,
    "(discrete action space)",
    ha="center", fontsize=6.2, color=DARK
)


# ---------------------------------------------------------------------
# PANEL 3 — CA-RDQN and CA-DQN
# ---------------------------------------------------------------------
ax.text(
    9.69, 6.72,
    "Four-step measurement and availability history",
    ha="center", fontsize=6.9, color=DARK
)

# History -> GRU -> DQN -> 4 sensors
rounded_box(8.03, 5.68, 0.80, 0.64, "#F7F3FB", PURPLE, 0.7)
ax.text(
    8.43, 6.00,
    "4-step\nhistory",
    ha="center", va="center",
    fontsize=6.5, color=DARK
)

arrow(8.88, 6.00, 9.10, 6.00, PURPLE, 1.5, 8)

rounded_box(9.10, 5.78, 0.50, 0.44, "#EEE6FA")
ax.text(
    9.35, 6.00, "GRU",
    ha="center", va="center",
    fontsize=7.7, fontweight="bold", color=PURPLE
)

arrow(9.66, 6.00, 9.86, 6.00, PURPLE, 1.5, 8)
neural_network(10.18, 6.00, 0.62)
arrow(10.66, 6.00, 10.92, 6.00, PURPLE, 1.5, 8)

rounded_box(10.92, 5.68, 0.50, 0.64, "#F7F3FB", PURPLE, 0.7)
ax.text(
    11.17, 6.00,
    "4\nsensors",
    ha="center", va="center",
    fontsize=6.5, fontweight="bold", color=DARK
)

ax.text(
    11.17, 5.54,
    "1 of 495 actions",
    ha="center", fontsize=5.8, color=DARK
)

# CA-DQN ablation
rounded_box(
    8.02, 3.43, 3.38, 1.72,
    "#F8F7FB", "#A7A0B4", 0.7
)

ax.text(
    9.71, 4.87,
    "CA-DQN (ablation)",
    ha="center", fontsize=9.0,
    fontweight="bold", color=PURPLE
)

ax.text(
    9.71, 4.63,
    "No recurrent layer",
    ha="center", fontsize=7.2, color=DARK
)

rounded_box(8.34, 3.91, 0.68, 0.46, "#F7F3FB", "#A7A0B4", 0.6)
ax.text(
    8.68, 4.14,
    "Current\nobservation",
    ha="center", va="center",
    fontsize=6.0, color=DARK
)

arrow(9.08, 4.14, 9.29, 4.14, PURPLE, 1.3, 8)
neural_network(9.68, 4.14, 0.50)
arrow(10.00, 4.14, 10.28, 4.14, PURPLE, 1.3, 8)

rounded_box(10.28, 3.91, 0.64, 0.46, "#F7F3FB", "#A7A0B4", 0.6)
ax.text(
    10.60, 4.14,
    "4 sensors",
    ha="center", va="center",
    fontsize=6.4, color=DARK
)

ax.text(
    10.60, 3.76,
    "1 of 495 actions",
    ha="center", fontsize=5.8, color=DARK
)

ax.text(
    9.71, 3.49,
    "Same budget, action space, estimator interface and evaluation protocol",
    ha="center", fontsize=5.5, color=GREY
)


# ---------------------------------------------------------------------
# PANEL 4 — Causal estimator and feedback
# ---------------------------------------------------------------------
rounded_box(12.00, 5.88, 0.94, 0.64, "#F7F7F7", "#AAB2BA", 0.7)
ax.text(
    12.47, 6.20,
    "Selected\nmeasurements",
    ha="center", va="center",
    fontsize=6.5, color=DARK
)

arrow(12.98, 6.20, 13.24, 6.20, GREY, 1.5, 8)

rounded_box(13.24, 5.88, 0.96, 0.64, "#F7F7F7", "#AAB2BA", 0.7)
ax.text(
    13.72, 6.20,
    "Causal\nestimator",
    ha="center", va="center",
    fontsize=6.7, color=DARK
)

arrow(14.25, 6.20, 14.53, 6.20, GREY, 1.5, 8)

rounded_box(14.53, 6.42, 1.05, 0.42, "#E5F1FA", BLUE, 0.7)
ax.text(
    15.055, 6.63,
    "ΔNH₄-N",
    ha="center", va="center",
    fontsize=7.7, fontweight="bold", color=DARK
)
ax.text(
    15.055, 6.48,
    "30-min ahead",
    ha="center", va="center",
    fontsize=5.8, color=DARK
)

rounded_box(14.53, 5.80, 1.05, 0.42, "#E5F3E7", GREEN, 0.7)
ax.text(
    15.055, 6.01,
    "ΔSNO",
    ha="center", va="center",
    fontsize=7.7, fontweight="bold", color=DARK
)
ax.text(
    15.055, 5.86,
    "30-min ahead",
    ha="center", va="center",
    fontsize=5.8, color=DARK
)

rounded_box(12.30, 4.88, 3.00, 0.52, "#FBE1DE", RED, 0.8)
ax.text(
    13.80, 5.14,
    "Estimation error → RL reward",
    ha="center", va="center",
    fontsize=7.5, fontweight="bold", color=DARK
)

arrow(13.80, 5.98, 13.80, 5.43, RED, 1.6, 8)

ax.text(
    13.80, 4.55,
    "Advance to next 15-min decision",
    ha="center", fontsize=7.5,
    color=NAVY, fontweight="bold"
)

ax.add_patch(
    FancyArrowPatch(
        (12.35, 4.74), (10.75, 3.13),
        connectionstyle="arc3,rad=.20",
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.8,
        color=BLUE
    )
)

ax.text(
    11.15, 3.26,
    "feedback",
    ha="center", fontsize=6.5,
    color=BLUE, fontweight="bold"
)

# Cross-panel arrows
arrow(3.85, 5.50, 3.99, 5.50, GREY, 2.8, 15)
arrow(7.69, 5.50, 7.82, 5.50, GREY, 2.8, 15)
arrow(11.59, 5.50, 11.72, 5.50, GREY, 2.8, 15)


# ---------------------------------------------------------------------
# KEY FINDINGS
# ---------------------------------------------------------------------
rounded_box(
    0.12, 1.34, 15.76, 1.48,
    "#E9F3FB", "#C7DCEB", 0.7
)

ax.text(
    0.40, 2.57,
    "Key findings",
    ha="left", va="center",
    fontsize=12.5, fontweight="bold", color=NAVY
)

for x in [4.0, 7.95, 11.86]:
    ax.plot([x, x], [1.55, 2.35], color="#BFC8D1", linewidth=0.8)

findings = [
    (0.55, BLUE, "Robust performance",
     "Consistent results across\n10 independent training seeds"),

    (4.35, GREEN, "Effective sensor selection",
     "Exactly 4 of 12 sensors\nwith 495 feasible configurations"),

    (8.30, ORANGE, "Temporal variation",
     "Informative sensor configurations\nvary across evaluation windows"),

    (12.22, PURPLE, "Recurrence ablation",
     "CA-RDQN and CA-DQN compared\nunder the same protocol"),
]

for x, colour, title, description in findings:
    ax.add_patch(
        Circle(
            (x + 0.40, 1.96),
            0.29,
            facecolor=colour,
            edgecolor="none"
        )
    )

    ax.text(
        x + 0.82, 2.10,
        title,
        ha="left", va="center",
        fontsize=8.5, fontweight="bold", color=NAVY
    )

    ax.text(
        x + 0.82, 1.77,
        description,
        ha="left", va="center",
        fontsize=6.9, color=DARK
    )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
rounded_box(0.12, 0.18, 15.76, 0.88, NAVY)

ax.text(
    8, 0.64,
    "Constraint-aware reinforcement learning for efficient, reliable and sustainable wastewater monitoring",
    ha="center", va="center",
    fontsize=10.7, fontweight="bold", color=WHITE
)

ax.text(
    15.35, 0.35,
    "CLEANER WATER  |  RESOURCE EFFICIENCY  |  SUSTAINABLE FUTURES",
    ha="right", va="center",
    fontsize=5.8, color=WHITE
)


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
png = OUT / "graphical_abstract_energy_reports.png"
pdf = OUT / "graphical_abstract_energy_reports.pdf"
svg = OUT / "graphical_abstract_energy_reports.svg"

fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor=WHITE)
fig.savefig(pdf, bbox_inches="tight", facecolor=WHITE)
fig.savefig(svg, bbox_inches="tight", facecolor=WHITE)

plt.close(fig)

print("\nGraphical abstract created successfully.")
print(f"PNG : {png}")
print(f"PDF : {pdf}")
print(f"SVG : {svg}")

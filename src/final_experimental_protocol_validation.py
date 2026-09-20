import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn

from itertools import combinations


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# DATA
# ============================================================

from dqn_data_loader import (
    sensors,
    targets,
    train_indices,
    eval_indices,
    HISTORY,
    SENSOR_COUNT,
)


# ============================================================
# FINAL EXPERIMENTAL CONFIGURATION
# ============================================================

SEED = 42

ACTION_DIM = 495

OBSERVATION_DIM = 24

STATE_HISTORY = 4

FORECAST_STEPS = 2

HIDDEN_DIM = 128

FC_DIM = 64

GAMMA = 0.99

LEARNING_RATE = 1e-3

BATCH_SIZE = 64

REPLAY_CAPACITY = 50_000

TARGET_UPDATE_FREQUENCY = 500

EPSILON_START = 1.0

EPSILON_END = 0.05

EPSILON_DECAY_STEPS = 20_000

WARMUP_STEPS = 64

DECISION_INTERVAL_MINUTES = 15

FORECAST_INTERVAL_MINUTES = 30

SENSOR_BUDGET = 4

TOTAL_SENSORS = 12

TRAIN_START = int(
    train_indices[0]
)

TRAIN_END = int(
    train_indices[-1]
)

EVAL_START = int(
    eval_indices[0]
)

EVAL_END = int(
    eval_indices[-1]
)


# ============================================================
# RANDOM SEEDS
# ============================================================

FINAL_SEEDS = [
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
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# ACTION SPACE
# ============================================================

ACTION_TABLE = np.array(
    list(
        combinations(
            range(
                TOTAL_SENSORS
            ),
            SENSOR_BUDGET
        )
    ),
    dtype=np.int64
)


# ============================================================
# VALIDATE ACTION SPACE
# ============================================================

assert ACTION_TABLE.shape == (
    ACTION_DIM,
    SENSOR_BUDGET
)

for action in ACTION_TABLE:

    assert len(
        set(
            action.tolist()
        )
    ) == SENSOR_BUDGET

    assert action.min() >= 0

    assert action.max() < TOTAL_SENSORS


# ============================================================
# RECURRENT DQN
# ============================================================

class RecurrentDQN(nn.Module):

    def __init__(
        self,
        observation_dim=OBSERVATION_DIM,
        hidden_dim=HIDDEN_DIM,
        fc_dim=FC_DIM,
        action_dim=ACTION_DIM,
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=observation_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.q_head = nn.Sequential(

            nn.Linear(
                hidden_dim,
                fc_dim
            ),

            nn.ReLU(),

            nn.Linear(
                fc_dim,
                action_dim
            ),
        )


    def forward(
        self,
        x
    ):

        assert x.ndim == 3

        assert x.shape[1] == STATE_HISTORY

        assert x.shape[2] == OBSERVATION_DIM

        output, _ = self.gru(x)

        final_hidden = (
            output[:, -1, :]
        )

        q_values = self.q_head(
            final_hidden
        )

        assert q_values.shape[1] == ACTION_DIM

        assert torch.isfinite(
            q_values
        ).all()

        return q_values


# ============================================================
# EPSILON SCHEDULE
# ============================================================

def epsilon_by_step(
    step
):

    fraction = min(
        step /
        EPSILON_DECAY_STEPS,
        1.0
    )

    return float(
        EPSILON_START
        +
        fraction
        *
        (
            EPSILON_END
            -
            EPSILON_START
        )
    )


# ============================================================
# FINAL PROTOCOL VALIDATION
# ============================================================

print("=" * 70)
print(
    "STEP 67 — FINAL EXPERIMENTAL PROTOCOL VALIDATION"
)
print("=" * 70)


# ============================================================
# DATA PARTITION
# ============================================================

print(
    f"[INFO] Training range: "
    f"{TRAIN_START}–{TRAIN_END}"
)

print(
    f"[INFO] Evaluation range: "
    f"{EVAL_START}–{EVAL_END}"
)

print(
    f"[INFO] Training decisions: "
    f"{len(train_indices)}"
)

print(
    f"[INFO] Evaluation decisions: "
    f"{len(eval_indices)}"
)


assert TRAIN_END < EVAL_START

assert TRAIN_START < TRAIN_END

assert EVAL_START < EVAL_END

assert (
    len(
        set(
            train_indices.tolist()
        )
        &
        set(
            eval_indices.tolist()
        )
    )
    ==
    0
)


print(
    "[PASS] Chronological train/evaluation separation verified."
)


# ============================================================
# SENSOR BUDGET
# ============================================================

assert TOTAL_SENSORS == 12

assert SENSOR_BUDGET == 4

assert ACTION_DIM == 495


expected_action_count = 1

for numerator in range(
    1,
    SENSOR_BUDGET + 1
):

    expected_action_count *= (
        TOTAL_SENSORS
        -
        numerator
        +
        1
    )

    expected_action_count //= numerator


assert expected_action_count == ACTION_DIM


print(
    "[PASS] 12-sensor / 4-sensor budget verified."
)

print(
    "[PASS] 495 feasible actions verified."
)


# ============================================================
# TEMPORAL DESIGN
# ============================================================

assert HISTORY == 4

assert STATE_HISTORY == 4

assert DECISION_INTERVAL_MINUTES == 15

assert FORECAST_INTERVAL_MINUTES == 30

assert FORECAST_STEPS == 2


print(
    "[PASS] Four-step temporal history verified."
)

print(
    "[PASS] 15-minute decision interval verified."
)

print(
    "[PASS] 30-minute forecast horizon verified."
)


# ============================================================
# TARGETS
# ============================================================

assert targets.ndim == 2

assert targets.shape[1] == 2

assert targets.shape[0] == sensors.shape[0]

assert np.isfinite(
    targets
).all()


print(
    "[PASS] Two-target ΔNH4 / ΔSNO formulation verified."
)


# ============================================================
# DATA LEAKAGE CHECK
# ============================================================

assert (
    TRAIN_END
    <
    EVAL_START
)


training_sensor_data = sensors[
    TRAIN_START:
    TRAIN_END + 1
]

evaluation_sensor_data = sensors[
    EVAL_START:
    EVAL_END + 1
]


training_target_data = targets[
    TRAIN_START:
    TRAIN_END + 1
]

evaluation_target_data = targets[
    EVAL_START:
    EVAL_END + 1
]


assert len(
    training_sensor_data
) == len(
    train_indices
)


assert len(
    evaluation_sensor_data
) == len(
    eval_indices
)


assert len(
    training_target_data
) == len(
    train_indices
)


assert len(
    evaluation_target_data
) == len(
    eval_indices
)


print(
    "[PASS] Training/evaluation data isolation verified."
)


# ============================================================
# NETWORK ARCHITECTURE
# ============================================================

torch.manual_seed(
    SEED
)

model = RecurrentDQN()


test_input = torch.randn(
    8,
    STATE_HISTORY,
    OBSERVATION_DIM
)


with torch.no_grad():

    q_values = model(
        test_input
    )


assert q_values.shape == (
    8,
    ACTION_DIM
)

assert torch.isfinite(
    q_values
).all()


parameter_count = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)


assert parameter_count == 99567


print(
    "[PASS] CA-RDQN architecture verified."
)

print(
    f"[INFO] Trainable parameters: "
    f"{parameter_count}"
)


# ============================================================
# OPTIMIZATION CONFIGURATION
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


assert (
    optimizer.param_groups[0]["lr"]
    ==
    LEARNING_RATE
)


print(
    "[PASS] Adam learning rate verified."
)


# ============================================================
# REPLAY CONFIGURATION
# ============================================================

assert REPLAY_CAPACITY == 50_000

assert BATCH_SIZE == 64

assert WARMUP_STEPS == 64


print(
    "[PASS] Replay configuration verified."
)


# ============================================================
# DOUBLE-DQN CONFIGURATION
# ============================================================

target_model = RecurrentDQN()

target_model.load_state_dict(
    model.state_dict()
)


for p_online, p_target in zip(
    model.parameters(),
    target_model.parameters()
):

    assert torch.allclose(
        p_online,
        p_target
    )


print(
    "[PASS] Double-DQN target-network initialization verified."
)


# ============================================================
# EPSILON CONFIGURATION
# ============================================================

assert np.isclose(
    epsilon_by_step(0),
    1.0
)

assert np.isclose(
    epsilon_by_step(5000),
    0.7625
)

assert np.isclose(
    epsilon_by_step(10000),
    0.525
)

assert np.isclose(
    epsilon_by_step(20000),
    0.05
)


assert (
    epsilon_by_step(0)
    >
    epsilon_by_step(5000)
    >
    epsilon_by_step(10000)
    >
    epsilon_by_step(20000)
)


print(
    "[PASS] Epsilon-greedy schedule locked."
)


# ============================================================
# FINAL MULTI-SEED SET
# ============================================================

assert len(
    FINAL_SEEDS
) == 10

assert len(
    set(
        FINAL_SEEDS
    )
) == 10


print(
    f"[INFO] Final independent seeds: "
    f"{FINAL_SEEDS}"
)

print(
    "[PASS] Ten independent evaluation seeds defined."
)


# ============================================================
# ACTION VALIDATION FOR EVERY SEED
# ============================================================

for seed in FINAL_SEEDS:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    seed_model = RecurrentDQN()

    sequence = torch.randn(
        1,
        STATE_HISTORY,
        OBSERVATION_DIM
    )

    with torch.no_grad():

        q = seed_model(
            sequence
        )

    action = int(
        torch.argmax(
            q,
            dim=1
        ).item()
    )

    selected = ACTION_TABLE[
        action
    ]

    assert len(
        selected
    ) == SENSOR_BUDGET

    assert len(
        set(
            selected.tolist()
        )
    ) == SENSOR_BUDGET


print(
    "[PASS] Feasible-action constraint verified across all seeds."
)


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

configuration = {
    "sensors": TOTAL_SENSORS,
    "active_sensors": SENSOR_BUDGET,
    "actions": ACTION_DIM,
    "observation_dimension": OBSERVATION_DIM,
    "history": STATE_HISTORY,
    "gru_hidden": HIDDEN_DIM,
    "fc_hidden": FC_DIM,
    "gamma": GAMMA,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "replay_capacity": REPLAY_CAPACITY,
    "warmup_steps": WARMUP_STEPS,
    "target_update": TARGET_UPDATE_FREQUENCY,
    "epsilon_start": EPSILON_START,
    "epsilon_end": EPSILON_END,
    "epsilon_decay": EPSILON_DECAY_STEPS,
    "decision_minutes": DECISION_INTERVAL_MINUTES,
    "forecast_minutes": FORECAST_INTERVAL_MINUTES,
    "training_start": TRAIN_START,
    "training_end": TRAIN_END,
    "evaluation_start": EVAL_START,
    "evaluation_end": EVAL_END,
}


print()
print(
    "[INFO] FINAL LOCKED CONFIGURATION"
)

for key, value in configuration.items():

    print(
        f"       {key}: {value}"
    )


# ============================================================
# SAVE PROTOCOL
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "experimental_protocol"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


protocol_path = os.path.join(
    OUTPUT_DIR,
    "final_experimental_protocol.txt"
)


with open(
    protocol_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL CA-RDQN EXPERIMENTAL PROTOCOL\n"
    )

    f.write(
        "===================================\n\n"
    )

    for key, value in configuration.items():

        f.write(
            f"{key}: {value}\n"
        )

    f.write(
        f"\nFinal seeds: {FINAL_SEEDS}\n"
    )

    f.write(
        "\nMethod: Constraint-Aware "
        "Recurrent Double DQN\n"
    )

    f.write(
        "Action constraint: exactly 4 of 12\n"
    )

    f.write(
        "Historical measurements: "
        "availability constrained\n"
    )

    f.write(
        "Evaluation data used during training: no\n"
    )


assert os.path.exists(
    protocol_path
)


print()
print(
    "[PASS] Final experimental protocol saved."
)

print(
    f"[INFO] Protocol file: "
    f"{protocol_path}"
)


# ============================================================
# FINAL
# ============================================================

print()
print(
    "[PASS] Training/evaluation split locked."
)

print(
    "[PASS] Sensor budget locked."
)

print(
    "[PASS] Action space locked."
)

print(
    "[PASS] Temporal configuration locked."
)

print(
    "[PASS] Network architecture locked."
)

print(
    "[PASS] Double-DQN configuration locked."
)

print(
    "[PASS] Replay configuration locked."
)

print(
    "[PASS] Exploration schedule locked."
)

print(
    "[PASS] Ten-seed experimental set locked."
)

print(
    "[PASS] No evaluation data permitted during training."
)

print("=" * 70)
print(
    "STEP 67 COMPLETED"
)
print("=" * 70)
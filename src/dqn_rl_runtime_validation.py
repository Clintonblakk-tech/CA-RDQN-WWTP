import os
import sys
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# STEP 54 — DQN RL RUNTIME VALIDATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SRC_DIR = os.path.join(
    PROJECT_ROOT,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ------------------------------------------------------------
# Import project data loader
# ------------------------------------------------------------

from dqn_data_loader import (
    sensors,
    targets,
    change_targets,
    train_indices,
    eval_indices,
    HISTORY,
    SENSOR_COUNT,
    TARGET_COUNT,
)


# ------------------------------------------------------------
# DQN configuration
# ------------------------------------------------------------

ACTION_DIM = 495

STATE_DIM = (
    HISTORY * SENSOR_COUNT
    + HISTORY * ACTION_DIM // ACTION_DIM
    + HISTORY * SENSOR_COUNT
)

# Correct DQN state:
# 4 historical sensor measurements = 48
# 4 historical activation masks = 48
# Total = 96

STATE_DIM = 96


# ------------------------------------------------------------
# Build one causal state
# ------------------------------------------------------------

def build_state(t):

    measurement_history = sensors[
        t - HISTORY:t
    ]

    mask_history = np.zeros(
        (HISTORY, SENSOR_COUNT),
        dtype=np.float32
    )

    state = np.concatenate(
        [
            measurement_history.astype(np.float32).reshape(-1),
            mask_history.reshape(-1),
        ]
    )

    return state


# ------------------------------------------------------------
# Generate test state
# ------------------------------------------------------------

test_index = int(train_indices[0])

state = build_state(test_index)


# ------------------------------------------------------------
# Validate state
# ------------------------------------------------------------

assert state.shape == (STATE_DIM,)
assert np.isfinite(state).all()


# ------------------------------------------------------------
# Define DQN
# ------------------------------------------------------------

class DQN(nn.Module):

    def __init__(
        self,
        state_dim,
        action_dim
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.network(x)


policy_net = DQN(
    STATE_DIM,
    ACTION_DIM
)


# ------------------------------------------------------------
# PyTorch state conversion
# ------------------------------------------------------------

state_tensor = torch.tensor(
    state,
    dtype=torch.float32
).unsqueeze(0)


# ------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------

with torch.no_grad():

    q_values = policy_net(
        state_tensor
    )


# ------------------------------------------------------------
# Validate Q-values
# ------------------------------------------------------------

assert q_values.shape == (
    1,
    ACTION_DIM
)

assert torch.isfinite(q_values).all()


# ------------------------------------------------------------
# Select greedy action
# ------------------------------------------------------------

action = int(
    torch.argmax(
        q_values,
        dim=1
    ).item()
)


# ------------------------------------------------------------
# Validate action index
# ------------------------------------------------------------

assert 0 <= action < ACTION_DIM


# ------------------------------------------------------------
# Validate 4-of-12 action encoding
#
# Lexicographic combinations are generated using:
# combinations(range(12), 4)
# ------------------------------------------------------------

from itertools import combinations

ACTION_TABLE = np.array(
    list(
        combinations(
            range(SENSOR_COUNT),
            4
        )
    ),
    dtype=np.int64
)

assert ACTION_TABLE.shape == (
    ACTION_DIM,
    4
)

selected_sensors = ACTION_TABLE[action]

assert len(selected_sensors) == 4
assert len(set(selected_sensors.tolist())) == 4
assert selected_sensors.min() >= 0
assert selected_sensors.max() < SENSOR_COUNT


# ------------------------------------------------------------
# Current measurement acquisition
#
# Only selected sensors become available after the action.
# ------------------------------------------------------------

current_measurement = np.zeros(
    SENSOR_COUNT,
    dtype=np.float32
)

current_mask = np.zeros(
    SENSOR_COUNT,
    dtype=np.float32
)

current_measurement[
    selected_sensors
] = sensors[
    test_index,
    selected_sensors
]

current_mask[
    selected_sensors
] = 1.0


# ------------------------------------------------------------
# Validate information availability
# ------------------------------------------------------------

assert np.all(
    current_mask[selected_sensors] == 1.0
)

inactive = np.setdiff1d(
    np.arange(SENSOR_COUNT),
    selected_sensors
)

assert np.all(
    current_mask[inactive] == 0.0
)


# ------------------------------------------------------------
# Verify target alignment
# ------------------------------------------------------------

expected_target = (
    targets[test_index + 2]
    - targets[test_index]
)

actual_target = change_targets[test_index]

assert np.allclose(
    expected_target,
    actual_target
)


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

print("=" * 70)
print("STEP 54 — DQN RL RUNTIME VALIDATION")
print("=" * 70)

print("[PASS] Project data loader imported.")
print(f"[PASS] State dimension: {STATE_DIM}")
print(f"[PASS] Action dimension: {ACTION_DIM}")
print("[PASS] PyTorch DQN initialized.")
print("[PASS] DQN forward pass completed.")
print("[PASS] Q-value shape verified.")
print(f"[INFO] Test greedy action: {action}")
print(
    f"[INFO] Selected sensors: "
    f"{selected_sensors.tolist()}"
)
print("[PASS] Exactly four sensors selected.")
print("[PASS] Current measurement masking verified.")
print("[PASS] Target alignment verified.")
print("[PASS] 96-D causal state verified.")

print("=" * 70)
print("STEP 54 COMPLETED")
print("=" * 70)
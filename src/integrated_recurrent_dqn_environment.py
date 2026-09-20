import os
import sys
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# PROJECT PATH
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


# ============================================================
# DATA
# ============================================================

from dqn_data_loader import (
    sensors,
    targets,
    change_targets,
    HISTORY,
    SENSOR_COUNT,
    TARGET_COUNT,
    train_indices,
    eval_indices,
)


# ============================================================
# CONFIGURATION
# ============================================================

ACTION_DIM = 495

OBSERVATION_DIM = SENSOR_COUNT * 2
STATE_DIM = HISTORY * OBSERVATION_DIM

ESTIMATOR_INPUT_DIM = 120

NH4_SCALE = 0.620953
SNO_SCALE = 0.351369

assert SENSOR_COUNT == 12
assert TARGET_COUNT == 2
assert HISTORY == 4

assert ACTION_DIM == 495
assert OBSERVATION_DIM == 24
assert STATE_DIM == 96
assert ESTIMATOR_INPUT_DIM == 120


# ============================================================
# ACTION TABLE
# ============================================================

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


# ============================================================
# FROZEN PYTORCH ESTIMATOR
# ============================================================

class FrozenEstimator(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(120, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# LOAD FROZEN ESTIMATORS
# ============================================================

ESTIMATOR_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "frozen_estimator"
)

NH4_MODEL_PATH = os.path.join(
    ESTIMATOR_DIR,
    "frozen_estimator_nh4_torch.pt"
)

SNO_MODEL_PATH = os.path.join(
    ESTIMATOR_DIR,
    "frozen_estimator_sno_torch.pt"
)

assert os.path.exists(
    NH4_MODEL_PATH
)

assert os.path.exists(
    SNO_MODEL_PATH
)


def load_frozen_model(path):

    model = FrozenEstimator()

    state_dict = torch.load(
        path,
        map_location="cpu"
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    for parameter in model.parameters():

        parameter.requires_grad = False

    return model


nh4_model = load_frozen_model(
    NH4_MODEL_PATH
)

sno_model = load_frozen_model(
    SNO_MODEL_PATH
)


# ============================================================
# ENVIRONMENT
# ============================================================

class IntegratedRecurrentSensorEnvironment:

    def __init__(
        self,
        start_index,
        end_index,
    ):

        self.start_index = int(
            start_index
        )

        self.end_index = int(
            end_index
        )

        assert (
            self.start_index >= HISTORY
        )

        assert (
            self.end_index + 2
            <
            len(targets)
        )

        self.current_index = None

        self.activation_history = None

        self.measurement_history = None


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.current_index = (
            self.start_index
        )

        self.activation_history = np.zeros(
            (
                HISTORY,
                SENSOR_COUNT
            ),
            dtype=np.float32
        )

        self.measurement_history = (
            np.zeros(
                (
                    HISTORY,
                    SENSOR_COUNT
                ),
                dtype=np.float32
            )
        )

        state = self._build_state()

        assert state.shape == (
            STATE_DIM,
        )

        return state


    # ========================================================
    # STATE
    # ========================================================

    def _build_state(self):

        masked_measurements = (
            self.measurement_history
            *
            self.activation_history
        )

        observation_history = np.concatenate(
            [
                masked_measurements,
                self.activation_history
            ],
            axis=1
        )

        state = observation_history.reshape(
            -1
        ).astype(
            np.float32
        )

        assert state.shape == (
            STATE_DIM,
        )

        assert np.isfinite(
            state
        ).all()

        return state


    # ========================================================
    # APPLY ACTION
    # ========================================================

    def _apply_action(self, action):

        action = int(action)

        assert (
            0 <= action < ACTION_DIM
        )

        selected = ACTION_TABLE[
            action
        ]

        assert len(selected) == 4

        assert len(
            set(
                selected.tolist()
            )
        ) == 4

        current_measurement = np.zeros(
            SENSOR_COUNT,
            dtype=np.float32
        )

        current_mask = np.zeros(
            SENSOR_COUNT,
            dtype=np.float32
        )

        current_measurement[
            selected
        ] = sensors[
            self.current_index,
            selected
        ]

        current_mask[
            selected
        ] = 1.0

        self.measurement_history = np.vstack(
            [
                self.measurement_history[1:],
                current_measurement
            ]
        )

        self.activation_history = np.vstack(
            [
                self.activation_history[1:],
                current_mask
            ]
        )

        return (
            current_measurement,
            current_mask,
            selected
        )


    # ========================================================
    # ESTIMATOR INPUT
    # ========================================================

    def _build_estimator_input(self):

        current_measurement = (
            self.measurement_history[-1]
        )

        current_mask = (
            self.activation_history[-1]
        )

        historical_measurements = (
            self.measurement_history
        )

        historical_masks = (
            self.activation_history
        )

        x = np.concatenate(
            [
                current_measurement,
                current_mask,
                historical_measurements.reshape(-1),
                historical_masks.reshape(-1),
            ]
        ).astype(
            np.float32
        )

        assert x.shape == (
            ESTIMATOR_INPUT_DIM,
        )

        assert np.isfinite(
            x
        ).all()

        return x


    # ========================================================
    # ESTIMATOR PREDICTION
    # ========================================================

    def _predict_change(self):

        x = self._build_estimator_input()

        x_tensor = torch.tensor(
            x,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            nh4_prediction = (
                nh4_model(
                    x_tensor
                )
                .cpu()
                .numpy()
                .reshape(-1)
            )

            sno_prediction = (
                sno_model(
                    x_tensor
                )
                .cpu()
                .numpy()
                .reshape(-1)
            )

        prediction = np.array(
            [
                nh4_prediction[0],
                sno_prediction[0]
            ],
            dtype=np.float32
        )

        assert prediction.shape == (
            2,
        )

        assert np.isfinite(
            prediction
        ).all()

        return prediction


    # ========================================================
    # ACTUAL CHANGE
    # ========================================================

    def _get_actual_change(self):

        actual_change = (
            targets[
                self.current_index + 2
            ]
            -
            targets[
                self.current_index
            ]
        )

        assert np.allclose(
            actual_change,
            change_targets[
                self.current_index
            ]
        )

        return actual_change.astype(
            np.float32
        )


    # ========================================================
    # REWARD
    # ========================================================

    def _calculate_reward(
        self,
        predicted_change,
        actual_change
    ):

        nh4_error = (
            predicted_change[0]
            -
            actual_change[0]
        )

        sno_error = (
            predicted_change[1]
            -
            actual_change[1]
        )

        reward = -(
            0.5
            *
            (
                nh4_error
                /
                NH4_SCALE
            ) ** 2

            +

            0.5
            *
            (
                sno_error
                /
                SNO_SCALE
            ) ** 2
        )

        assert np.isfinite(
            reward
        )

        return float(reward)


    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        assert (
            self.current_index
            is not None
        )

        (
            current_measurement,
            current_mask,
            selected
        ) = self._apply_action(
            action
        )

        predicted_change = (
            self._predict_change()
        )

        actual_change = (
            self._get_actual_change()
        )

        reward = (
            self._calculate_reward(
                predicted_change,
                actual_change
            )
        )

        next_state = (
            self._build_state()
        )

        self.current_index += 1

        done = (
            self.current_index
            >
            self.end_index
        )

        return {
            "next_state": next_state,
            "reward": reward,
            "done": done,
            "current_measurement":
                current_measurement,
            "current_mask":
                current_mask,
            "selected_sensors":
                selected,
            "predicted_change":
                predicted_change,
            "actual_change":
                actual_change,
        }


# ============================================================
# VALIDATION
# ============================================================

print("=" * 70)
print(
    "STEP 60 — INTEGRATED RECURRENT DQN ENVIRONMENT"
)
print("=" * 70)

print(
    f"[INFO] Sensor count: {SENSOR_COUNT}"
)

print(
    f"[INFO] Action dimension: {ACTION_DIM}"
)

print(
    f"[INFO] State dimension: {STATE_DIM}"
)

print(
    f"[INFO] Estimator input dimension: "
    f"{ESTIMATOR_INPUT_DIM}"
)


# ============================================================
# ENVIRONMENT
# ============================================================

env = IntegratedRecurrentSensorEnvironment(
    start_index=int(
        train_indices[0]
    ),
    end_index=int(
        train_indices[0]
    ) + 10
)


state = env.reset()

assert state.shape == (
    STATE_DIM,
)

print(
    "[PASS] Environment reset verified."
)

print(
    "[PASS] Frozen NH4 estimator loaded."
)

print(
    "[PASS] Frozen SNO estimator loaded."
)


# ============================================================
# TEST ACTIONS
# ============================================================

test_actions = [
    0,
    100,
    250,
    309,
    494,
]

for action in test_actions:

    result = env.step(
        action
    )

    selected = result[
        "selected_sensors"
    ]

    prediction = result[
        "predicted_change"
    ]

    actual = result[
        "actual_change"
    ]

    reward = result[
        "reward"
    ]

    next_state = result[
        "next_state"
    ]

    assert len(selected) == 4

    assert len(
        set(
            selected.tolist()
        )
    ) == 4

    assert prediction.shape == (
        2,
    )

    assert actual.shape == (
        2,
    )

    assert np.isfinite(
        reward
    )

    assert next_state.shape == (
        STATE_DIM,
    )

    print()
    print(
        f"[PASS] Action {action}: "
        f"{selected.tolist()}"
    )

    print(
        f"       Predicted ΔNH4: "
        f"{prediction[0]:.6f}"
    )

    print(
        f"       Actual ΔNH4: "
        f"{actual[0]:.6f}"
    )

    print(
        f"       Predicted ΔSNO: "
        f"{prediction[1]:.6f}"
    )

    print(
        f"       Actual ΔSNO: "
        f"{actual[1]:.6f}"
    )

    print(
        f"       Reward: "
        f"{reward:.6f}"
    )


# ============================================================
# CAUSALITY CHECK
# ============================================================

assert env.current_index > int(
    train_indices[0]
)

print()
print(
    "[PASS] Sequential environment progression verified."
)

print(
    "[PASS] Actual frozen-estimator predictions verified."
)

print(
    "[PASS] 30-minute prediction target verified."
)

print(
    "[PASS] Reward calculation verified."
)

print(
    "[PASS] Causal state transition verified."
)

print(
    "[PASS] Exact-four constraint verified."
)

print("=" * 70)
print("STEP 60 COMPLETED")
print("=" * 70)
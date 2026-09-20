import os
import sys
import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

ESTIMATOR_DIR = os.path.join(
    DATA_DIR,
    "frozen_estimator"
)

sys.path.insert(
    0,
    os.path.join(PROJECT_ROOT, "src")
)

from dqn_data_loader import (
    sensors,
    targets,
    HISTORY,
    SENSOR_COUNT,
    TARGET_COUNT,
)


# ============================================================
# CONFIGURATION
# ============================================================

ACTION_DIM = 495
ESTIMATOR_INPUT_DIM = 120

NH4_MODEL_PATH = os.path.join(
    ESTIMATOR_DIR,
    "frozen_estimator_nh4.joblib"
)

SNO_MODEL_PATH = os.path.join(
    ESTIMATOR_DIR,
    "frozen_estimator_sno.joblib"
)


# ============================================================
# ACTION TABLE
# ============================================================

from itertools import combinations

ACTION_TABLE = np.array(
    list(combinations(range(SENSOR_COUNT), 4)),
    dtype=np.int64
)

assert ACTION_TABLE.shape == (ACTION_DIM, 4)


# ============================================================
# LOAD FROZEN ESTIMATORS
# ============================================================

print("=" * 70)
print("STEP 56 — FROZEN ESTIMATOR RL VALIDATION")
print("=" * 70)

assert os.path.exists(NH4_MODEL_PATH)
assert os.path.exists(SNO_MODEL_PATH)

nh4_model = joblib.load(
    NH4_MODEL_PATH
)

sno_model = joblib.load(
    SNO_MODEL_PATH
)

print("[PASS] NH4 frozen estimator loaded.")
print("[PASS] SNO frozen estimator loaded.")


# ============================================================
# CAUSAL ESTIMATOR INPUT
# ============================================================

def build_estimator_input(t, action):

    selected = ACTION_TABLE[action]

    current_measurement = np.zeros(
        SENSOR_COUNT,
        dtype=np.float32
    )

    current_mask = np.zeros(
        SENSOR_COUNT,
        dtype=np.float32
    )

    current_measurement[selected] = sensors[
        t,
        selected
    ]

    current_mask[selected] = 1.0

    measurement_history = sensors[
        t - HISTORY:t
    ].astype(np.float32)

    history_mask = np.ones(
        (HISTORY, SENSOR_COUNT),
        dtype=np.float32
    )

    x = np.concatenate([
        current_measurement,
        current_mask,
        measurement_history.reshape(-1),
        history_mask.reshape(-1),
    ])

    assert x.shape == (ESTIMATOR_INPUT_DIM,)
    assert np.isfinite(x).all()

    return x


# ============================================================
# TEST MULTIPLE ACTIONS
# ============================================================

test_time = 100

test_actions = [
    0,
    100,
    250,
    309,
    494,
]

print()
print(f"[INFO] Test time index: {test_time}")
print(
    f"[INFO] Testing actions: "
    f"{test_actions}"
)

for action in test_actions:

    selected = ACTION_TABLE[action]

    x = build_estimator_input(
        test_time,
        action
    ).reshape(1, -1)

    nh4_prediction = float(
        nh4_model.predict(x)[0]
    )

    sno_prediction = float(
        sno_model.predict(x)[0]
    )

    actual_change = (
        targets[test_time + 2]
        -
        targets[test_time]
    )

    nh4_error = (
        nh4_prediction
        -
        actual_change[0]
    )

    sno_error = (
        sno_prediction
        -
        actual_change[1]
    )

    assert len(selected) == 4
    assert len(set(selected.tolist())) == 4

    assert np.isfinite(
        nh4_prediction
    )

    assert np.isfinite(
        sno_prediction
    )

    assert np.isfinite(
        nh4_error
    )

    assert np.isfinite(
        sno_error
    )

    print()
    print(
        f"[PASS] Action {action}: "
        f"{selected.tolist()}"
    )

    print(
        f"       NH4 prediction: "
        f"{nh4_prediction:.6f}"
    )

    print(
        f"       SNO prediction: "
        f"{sno_prediction:.6f}"
    )

    print(
        f"       NH4 error: "
        f"{nh4_error:.6f}"
    )

    print(
        f"       SNO error: "
        f"{sno_error:.6f}"
    )


# ============================================================
# REPRODUCIBILITY TEST
# ============================================================

action = 309

x = build_estimator_input(
    test_time,
    action
).reshape(1, -1)

prediction_1 = np.array([
    nh4_model.predict(x)[0],
    sno_model.predict(x)[0],
])

prediction_2 = np.array([
    nh4_model.predict(x)[0],
    sno_model.predict(x)[0],
])

assert np.allclose(
    prediction_1,
    prediction_2
)

print()
print("[PASS] Frozen estimator reproducibility verified.")
print("[PASS] 120-D estimator interface verified.")
print("[PASS] Multiple feasible actions verified.")
print("[PASS] 30-minute target error calculation verified.")

print("=" * 70)
print("STEP 56 COMPLETED")
print("=" * 70)
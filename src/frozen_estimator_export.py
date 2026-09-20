import os
import sys
import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dqn_data_loader import (
    sensors,
    targets,
    train_indices,
    eval_indices,
    HISTORY,
    SENSOR_COUNT,
    TARGET_COUNT,
)

from itertools import combinations


# ============================================================
# CONFIGURATION
# ============================================================

ACTION_DIM = 495
ESTIMATOR_INPUT_DIM = 120

RANDOM_STATE = 42

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "frozen_estimator",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# ACTION TABLE
# ============================================================

ACTION_TABLE = np.array(
    list(combinations(range(SENSOR_COUNT), 4)),
    dtype=np.int64
)

assert ACTION_TABLE.shape == (ACTION_DIM, 4)


# ============================================================
# CAUSAL ESTIMATOR INPUT
# ============================================================

def build_estimator_input(t, action):
    """
    Construct the causal 120-D estimator input.

    Components:
        12 current masked measurements
        12 current availability mask
        48 historical measurements
        48 historical availability masks
    """

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

    # Historical availability is represented here as fully
    # available for the frozen-estimator construction.
    #
    # The dynamic DQN environment will replace this with
    # the actual historical activation masks.
    history_mask = np.ones(
        (HISTORY, SENSOR_COUNT),
        dtype=np.float32
    )

    state = np.concatenate([
        current_measurement,
        current_mask,
        measurement_history.reshape(-1),
        history_mask.reshape(-1),
    ])

    assert state.shape == (ESTIMATOR_INPUT_DIM,)

    return state


# ============================================================
# TRAINING SAMPLE CONSTRUCTION
# ============================================================

# Use representative training times so that all 495 actions
# are represented during estimator training.

candidate_indices = np.linspace(
    int(train_indices[0]),
    int(train_indices[-1]),
    20,
    dtype=int
)

candidate_indices = np.unique(candidate_indices)

X = []
Y_NH4 = []
Y_SNO = []

for action in range(ACTION_DIM):

    for t in candidate_indices:

        if t + 2 >= len(targets):
            continue

        x = build_estimator_input(t, action)

        delta_target = (
            targets[t + 2] -
            targets[t]
        )

        X.append(x)
        Y_NH4.append(delta_target[0])
        Y_SNO.append(delta_target[1])


X = np.asarray(X, dtype=np.float32)
Y_NH4 = np.asarray(Y_NH4, dtype=np.float32)
Y_SNO = np.asarray(Y_SNO, dtype=np.float32)


# ============================================================
# VALIDATION
# ============================================================

assert X.shape[1] == ESTIMATOR_INPUT_DIM
assert X.shape[0] == ACTION_DIM * len(candidate_indices)

assert Y_NH4.shape[0] == X.shape[0]
assert Y_SNO.shape[0] == X.shape[0]

assert np.isfinite(X).all()
assert np.isfinite(Y_NH4).all()
assert np.isfinite(Y_SNO).all()


# ============================================================
# TRAIN FROZEN ESTIMATORS
# ============================================================

nh4_model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    random_state=RANDOM_STATE,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)

sno_model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    random_state=RANDOM_STATE,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)


print("=" * 70)
print("STEP 55 — FROZEN ESTIMATOR EXPORT")
print("=" * 70)

print(f"[INFO] Action space: {ACTION_DIM}")
print(f"[INFO] Estimator input dimension: {ESTIMATOR_INPUT_DIM}")
print(f"[INFO] Candidate training times: {len(candidate_indices)}")
print(f"[INFO] Training examples: {len(X)}")

print()
print("[INFO] Training NH4 estimator...")

nh4_model.fit(
    X,
    Y_NH4
)

print("[PASS] NH4 estimator trained.")

print()
print("[INFO] Training SNO estimator...")

sno_model.fit(
    X,
    Y_SNO
)

print("[PASS] SNO estimator trained.")


# ============================================================
# FREEZE / SAVE
# ============================================================

nh4_path = os.path.join(
    OUTPUT_DIR,
    "frozen_estimator_nh4.joblib"
)

sno_path = os.path.join(
    OUTPUT_DIR,
    "frozen_estimator_sno.joblib"
)

metadata_path = os.path.join(
    OUTPUT_DIR,
    "frozen_estimator_metadata.txt"
)

joblib.dump(
    nh4_model,
    nh4_path
)

joblib.dump(
    sno_model,
    sno_path
)


# ============================================================
# SAVE METADATA
# ============================================================

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Frozen estimator metadata\n"
        "=========================\n"
    )

    f.write(
        f"Action dimension: {ACTION_DIM}\n"
    )

    f.write(
        f"Sensor count: {SENSOR_COUNT}\n"
    )

    f.write(
        f"History length: {HISTORY}\n"
    )

    f.write(
        f"Estimator input dimension: {ESTIMATOR_INPUT_DIM}\n"
    )

    f.write(
        "Hidden layers: (128, 64)\n"
    )

    f.write(
        "Activation: relu\n"
    )

    f.write(
        "Solver: adam\n"
    )

    f.write(
        f"Random state: {RANDOM_STATE}\n"
    )

    f.write(
        f"Training examples: {len(X)}\n"
    )

    f.write(
        f"Training times: {len(candidate_indices)}\n"
    )

    f.write(
        "Targets: Delta_NH4, Delta_SNO\n"
    )


# ============================================================
# RELOAD TEST
# ============================================================

loaded_nh4 = joblib.load(nh4_path)
loaded_sno = joblib.load(sno_path)

test_action = 309
test_time = int(train_indices[0])

test_x = build_estimator_input(
    test_time,
    test_action
).reshape(1, -1)

prediction_nh4 = loaded_nh4.predict(test_x)
prediction_sno = loaded_sno.predict(test_x)

assert prediction_nh4.shape == (1,)
assert prediction_sno.shape == (1,)

assert np.isfinite(prediction_nh4).all()
assert np.isfinite(prediction_sno).all()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("[PASS] NH4 estimator exported.")
print("[PASS] SNO estimator exported.")
print("[PASS] Frozen estimator reload verified.")
print(f"[INFO] Test action: {test_action}")
print(f"[INFO] Test time index: {test_time}")
print(
    f"[INFO] Test NH4 prediction: "
    f"{prediction_nh4[0]:.6f}"
)
print(
    f"[INFO] Test SNO prediction: "
    f"{prediction_sno[0]:.6f}"
)

print()
print(f"[INFO] Output directory: {OUTPUT_DIR}")
print(f"[INFO] NH4 model: {nh4_path}")
print(f"[INFO] SNO model: {sno_path}")
print(f"[INFO] Metadata: {metadata_path}")

print("=" * 70)
print("STEP 55 COMPLETED")
print("=" * 70)
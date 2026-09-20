import os
import sys
import random

import numpy as np
from itertools import combinations

from sklearn.neural_network import MLPRegressor


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
# DATA LOADER
# ============================================================

from dqn_data_loader import (
    sensors,
    targets,
    train_indices,
    HISTORY,
    SENSOR_COUNT,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

ACTION_DIM = 495

TRAIN_TIMES_PER_ACTION = 40

RANDOM_STATE = 42

HIDDEN_LAYERS = (128, 64)

MAX_ITER = 300

FORECAST_STEPS = 2


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "causal_dynamic_estimator"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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

assert np.all(
    [
        len(set(action.tolist())) == 4
        for action in ACTION_TABLE
    ]
)


# ============================================================
# ESTIMATOR INPUT
# ============================================================

def build_estimator_input(
    t,
    historical_actions,
    current_action,
):

    assert t >= HISTORY

    assert len(
        historical_actions
    ) == HISTORY


    # --------------------------------------------------------
    # Current measurements
    # --------------------------------------------------------

    current_measurements = np.zeros(
        SENSOR_COUNT,
        dtype=np.float32
    )

    current_mask = np.zeros(
        SENSOR_COUNT,
        dtype=np.float32
    )


    current_selected = ACTION_TABLE[
        current_action
    ]


    current_measurements[
        current_selected
    ] = sensors[
        t,
        current_selected
    ]


    current_mask[
        current_selected
    ] = 1.0


    # --------------------------------------------------------
    # Historical measurements
    # --------------------------------------------------------

    historical_measurements = np.zeros(
        (
            HISTORY,
            SENSOR_COUNT
        ),
        dtype=np.float32
    )

    historical_masks = np.zeros(
        (
            HISTORY,
            SENSOR_COUNT
        ),
        dtype=np.float32
    )


    for h in range(
        HISTORY
    ):

        history_t = (
            t
            -
            HISTORY
            +
            h
        )


        historical_action = int(
            historical_actions[h]
        )


        selected = ACTION_TABLE[
            historical_action
        ]


        historical_measurements[
            h,
            selected
        ] = sensors[
            history_t,
            selected
        ]


        historical_masks[
            h,
            selected
        ] = 1.0


    # --------------------------------------------------------
    # Final 120-D interface
    # --------------------------------------------------------

    x = np.concatenate(
        [
            current_measurements,
            current_mask,
            historical_measurements.reshape(-1),
            historical_masks.reshape(-1),
        ]
    ).astype(
        np.float32
    )


    assert x.shape == (
        120,
    )

    assert np.isfinite(
        x
    ).all()


    return x


# ============================================================
# TARGET
# ============================================================

def build_target(t):

    y = (
        targets[
            t + FORECAST_STEPS
        ]
        -
        targets[t]
    )

    y = np.asarray(
        y,
        dtype=np.float32
    )

    assert y.shape == (
        2,
    )

    assert np.isfinite(
        y
    ).all()

    return y


# ============================================================
# TRAINING TIME CANDIDATES
# ============================================================

training_times = np.asarray(
    train_indices,
    dtype=np.int64
)

training_times = training_times[
    training_times >= HISTORY
]

assert len(
    training_times
) >= TRAIN_TIMES_PER_ACTION


# ============================================================
# BUILD TRAINING DATA
# ============================================================

print("=" * 70)
print(
    "STEP 64 — CAUSAL DYNAMIC ESTIMATOR EXPORT"
)
print("=" * 70)

print(
    f"[INFO] Sensor count: {SENSOR_COUNT}"
)

print(
    f"[INFO] Action dimension: {ACTION_DIM}"
)

print(
    f"[INFO] History length: {HISTORY}"
)

print(
    "[INFO] Estimator input dimension: 120"
)

print(
    f"[INFO] Training times per action: "
    f"{TRAIN_TIMES_PER_ACTION}"
)

print()
print(
    "[INFO] Reconstructing causal training dataset..."
)


X_train = []

y_nh4 = []

y_sno = []


for action_id in range(
    ACTION_DIM
):

    selected_times = np.random.choice(
        training_times,
        size=TRAIN_TIMES_PER_ACTION,
        replace=False
    )


    for t in selected_times:

        historical_actions = np.random.randint(
            0,
            ACTION_DIM,
            size=HISTORY
        )


        x = build_estimator_input(
            int(t),
            historical_actions,
            action_id
        )


        y = build_target(
            int(t)
        )


        X_train.append(
            x
        )

        y_nh4.append(
            y[0]
        )

        y_sno.append(
            y[1]
        )


X_train = np.asarray(
    X_train,
    dtype=np.float32
)

y_nh4 = np.asarray(
    y_nh4,
    dtype=np.float32
)

y_sno = np.asarray(
    y_sno,
    dtype=np.float32
)


EXPECTED = (
    ACTION_DIM
    *
    TRAIN_TIMES_PER_ACTION
)


assert X_train.shape == (
    EXPECTED,
    120
)

assert y_nh4.shape == (
    EXPECTED,
)

assert y_sno.shape == (
    EXPECTED,
)


print(
    f"[PASS] Training matrix: "
    f"{X_train.shape}"
)


# ============================================================
# CAUSALITY CHECK
# ============================================================

historical_measurements = X_train[
    :,
    24:72
].reshape(
    -1,
    HISTORY,
    SENSOR_COUNT
)

historical_masks = X_train[
    :,
    72:120
].reshape(
    -1,
    HISTORY,
    SENSOR_COUNT
)


check_count = min(
    1000,
    len(X_train)
)


for i in range(
    check_count
):

    unavailable = (
        historical_masks[i] == 0
    )

    assert np.all(
        historical_measurements[i][
            unavailable
        ] == 0
    )


print(
    "[PASS] Causal historical availability verified."
)


# ============================================================
# TRAIN NH4
# ============================================================

print()
print(
    "[INFO] Training causal NH4 estimator..."
)


nh4_model = MLPRegressor(
    hidden_layer_sizes=HIDDEN_LAYERS,
    activation="relu",
    solver="adam",
    learning_rate_init=1e-3,
    max_iter=MAX_ITER,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
)


nh4_model.fit(
    X_train,
    y_nh4
)


assert nh4_model.n_features_in_ == 120

assert np.isfinite(
    nh4_model.loss_
)


print(
    f"[PASS] NH4 estimator trained "
    f"({nh4_model.n_iter_} iterations)."
)


# ============================================================
# TRAIN SNO
# ============================================================

print(
    "[INFO] Training causal SNO estimator..."
)


sno_model = MLPRegressor(
    hidden_layer_sizes=HIDDEN_LAYERS,
    activation="relu",
    solver="adam",
    learning_rate_init=1e-3,
    max_iter=MAX_ITER,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
)


sno_model.fit(
    X_train,
    y_sno
)


assert sno_model.n_features_in_ == 120

assert np.isfinite(
    sno_model.loss_
)


print(
    f"[PASS] SNO estimator trained "
    f"({sno_model.n_iter_} iterations)."
)


# ============================================================
# EXPORT NUMERICAL PARAMETERS
# ============================================================

def export_mlp(
    model,
    path,
):

    assert len(
        model.coefs_
    ) == 3

    assert len(
        model.intercepts_
    ) == 3


    W1 = model.coefs_[0].astype(
        np.float32
    )

    b1 = model.intercepts_[0].astype(
        np.float32
    )

    W2 = model.coefs_[1].astype(
        np.float32
    )

    b2 = model.intercepts_[1].astype(
        np.float32
    )

    W3 = model.coefs_[2].astype(
        np.float32
    )

    b3 = model.intercepts_[2].astype(
        np.float32
    )


    assert W1.shape == (
        120,
        128
    )

    assert b1.shape == (
        128,
    )

    assert W2.shape == (
        128,
        64
    )

    assert b2.shape == (
        64,
    )

    assert W3.shape == (
        64,
        1
    )

    assert b3.shape == (
        1,
    )


    for array in [
        W1,
        b1,
        W2,
        b2,
        W3,
        b3,
    ]:

        assert np.isfinite(
            array
        ).all()


    np.savez(
        path,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        W3=W3,
        b3=b3,
    )


    return (
        W1,
        b1,
        W2,
        b2,
        W3,
        b3,
    )


# ============================================================
# EXPORT
# ============================================================

nh4_path = os.path.join(
    OUTPUT_DIR,
    "causal_estimator_nh4_weights.npz"
)

sno_path = os.path.join(
    OUTPUT_DIR,
    "causal_estimator_sno_weights.npz"
)


nh4_parameters = export_mlp(
    nh4_model,
    nh4_path
)

sno_parameters = export_mlp(
    sno_model,
    sno_path
)


print()
print(
    "[PASS] NH4 numerical weights exported."
)

print(
    "[PASS] SNO numerical weights exported."
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata_path = os.path.join(
    OUTPUT_DIR,
    "causal_estimator_metadata.txt"
)


with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Causal Dynamic Estimator\n"
    )

    f.write(
        "=======================\n"
    )

    f.write(
        f"Sensor count: {SENSOR_COUNT}\n"
    )

    f.write(
        f"Action dimension: {ACTION_DIM}\n"
    )

    f.write(
        f"History length: {HISTORY}\n"
    )

    f.write(
        "Estimator input dimension: 120\n"
    )

    f.write(
        "Hidden layers: 128, 64\n"
    )

    f.write(
        "Activation: ReLU\n"
    )

    f.write(
        "Output activation: identity\n"
    )

    f.write(
        f"Training examples: {len(X_train)}\n"
    )

    f.write(
        f"Training times per action: "
        f"{TRAIN_TIMES_PER_ACTION}\n"
    )

    f.write(
        "Forecast horizon: 30 minutes\n"
    )

    f.write(
        "Random seed: 42\n"
    )

    f.write(
        "Historical measurements restricted "
        "by activation masks: yes\n"
    )


print(
    "[PASS] Metadata exported."
)


# ============================================================
# RELOAD VERIFICATION
# ============================================================

nh4_loaded = np.load(
    nh4_path
)

sno_loaded = np.load(
    sno_path
)


for key in [
    "W1",
    "b1",
    "W2",
    "b2",
    "W3",
    "b3",
]:

    assert np.array_equal(
        nh4_loaded[key],
        nh4_parameters[
            [
                "W1",
                "b1",
                "W2",
                "b2",
                "W3",
                "b3",
            ].index(key)
        ]
    )

    assert np.array_equal(
        sno_loaded[key],
        sno_parameters[
            [
                "W1",
                "b1",
                "W2",
                "b2",
                "W3",
                "b3",
            ].index(key)
        ]
    )


print(
    "[PASS] Exported parameter reload verified."
)


# ============================================================
# FORWARD-PREDICTION PRESERVATION
# ============================================================

test_times = training_times[
    :10
]

test_actions = np.arange(
    10
)


test_inputs = []

for t, action_id in zip(
    test_times,
    test_actions
):

    historical_actions = (
        np.arange(HISTORY)
        +
        action_id
    ) % ACTION_DIM

    test_inputs.append(
        build_estimator_input(
            int(t),
            historical_actions,
            int(action_id)
        )
    )


test_inputs = np.asarray(
    test_inputs,
    dtype=np.float32
)


original_nh4 = nh4_model.predict(
    test_inputs
)

original_sno = sno_model.predict(
    test_inputs
)


def manual_forward(
    parameters,
    X,
):

    W1, b1, W2, b2, W3, b3 = (
        parameters
    )

    h1 = np.maximum(
        0,
        X @ W1 + b1
    )

    h2 = np.maximum(
        0,
        h1 @ W2 + b2
    )

    output = (
        h2 @ W3
        +
        b3
    )

    return output.reshape(-1)


manual_nh4 = manual_forward(
    nh4_parameters,
    test_inputs
)

manual_sno = manual_forward(
    sno_parameters,
    test_inputs
)


assert np.allclose(
    original_nh4,
    manual_nh4,
    atol=1e-5
)

assert np.allclose(
    original_sno,
    manual_sno,
    atol=1e-5
)


print(
    "[PASS] NH4 forward-prediction preservation verified."
)

print(
    "[PASS] SNO forward-prediction preservation verified."
)


# ============================================================
# FINAL VALIDATION
# ============================================================

assert os.path.exists(
    nh4_path
)

assert os.path.exists(
    sno_path
)

assert os.path.exists(
    metadata_path
)


print()
print(
    "[PASS] Causal estimator architecture preserved."
)

print(
    "[PASS] Causal activation-history interface preserved."
)

print(
    "[PASS] 120-D input preserved."
)

print(
    "[PASS] Numerical parameter export verified."
)

print(
    "[PASS] Prediction equivalence verified."
)

print()
print(
    f"[INFO] Output directory:"
)

print(
    OUTPUT_DIR
)

print("=" * 70)
print(
    "STEP 64 COMPLETED"
)
print("=" * 70)
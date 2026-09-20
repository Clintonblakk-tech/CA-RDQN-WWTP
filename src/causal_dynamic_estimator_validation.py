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
    eval_indices,
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


print("=" * 70)
print(
    "STEP 63 — CAUSAL DYNAMIC ESTIMATOR VALIDATION"
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
    f"[INFO] Estimator input dimension: "
    f"{12 + 12 + 48 + 48}"
)

print(
    f"[INFO] Training times per action: "
    f"{TRAIN_TIMES_PER_ACTION}"
)


# ============================================================
# ESTIMATOR INPUT CONSTRUCTION
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

    current_sensors = np.zeros(
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

    current_sensors[
        current_selected
    ] = sensors[
        t,
        current_selected
    ]

    current_mask[
        current_selected
    ] = 1.0


    # --------------------------------------------------------
    # Historical measurements and availability
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


    for h in range(HISTORY):

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
    # 120-D interface
    # --------------------------------------------------------

    estimator_input = np.concatenate(
        [
            current_sensors,
            current_mask,
            historical_measurements.reshape(-1),
            historical_masks.reshape(-1),
        ]
    ).astype(
        np.float32
    )


    assert estimator_input.shape == (
        120,
    )

    assert np.isfinite(
        estimator_input
    ).all()

    return estimator_input


# ============================================================
# TARGET CONSTRUCTION
# ============================================================

def build_target(t):

    target = (
        targets[t + FORECAST_STEPS]
        -
        targets[t]
    )

    target = np.asarray(
        target,
        dtype=np.float32
    )

    assert target.shape == (
        2,
    )

    assert np.isfinite(
        target
    ).all()

    return target


# ============================================================
# GENERATE CAUSAL TRAINING DATA
# ============================================================

print()
print(
    "[INFO] Generating causally available "
    "dynamic activation histories..."
)


training_time_candidates = np.asarray(
    train_indices,
    dtype=np.int64
)

training_time_candidates = (
    training_time_candidates[
        training_time_candidates
        >= HISTORY
    ]
)

assert len(
    training_time_candidates
) > 0


X_train = []

y_train_nh4 = []

y_train_sno = []


# ------------------------------------------------------------
# Every feasible current action is represented.
# Historical actions are independently sampled from
# the feasible action set.
# ------------------------------------------------------------

for action_id in range(
    ACTION_DIM
):

    selected_training_times = np.random.choice(
        training_time_candidates,
        size=TRAIN_TIMES_PER_ACTION,
        replace=False
    )


    for t in selected_training_times:

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

        y_train_nh4.append(
            y[0]
        )

        y_train_sno.append(
            y[1]
        )


X_train = np.asarray(
    X_train,
    dtype=np.float32
)

y_train_nh4 = np.asarray(
    y_train_nh4,
    dtype=np.float32
)

y_train_sno = np.asarray(
    y_train_sno,
    dtype=np.float32
)


EXPECTED_TRAINING_EXAMPLES = (
    ACTION_DIM
    *
    TRAIN_TIMES_PER_ACTION
)


assert X_train.shape == (
    EXPECTED_TRAINING_EXAMPLES,
    120
)

assert y_train_nh4.shape == (
    EXPECTED_TRAINING_EXAMPLES,
)

assert y_train_sno.shape == (
    EXPECTED_TRAINING_EXAMPLES,
)


print(
    f"[INFO] Training examples: "
    f"{len(X_train)}"
)

print(
    f"[INFO] Training matrix shape: "
    f"{X_train.shape}"
)


# ============================================================
# VERIFY CURRENT ACTION COVERAGE
# ============================================================

# Current action is encoded in the current availability mask.
# Recovering the exact action is possible because exactly four
# sensors are active.

current_masks = X_train[
    :,
    12:24
]

unique_action_masks = np.unique(
    current_masks,
    axis=0
)


assert unique_action_masks.shape[0] == (
    ACTION_DIM
)


print(
    "[PASS] All 495 current activation "
    "configurations represented."
)


# ============================================================
# VERIFY HISTORICAL CAUSALITY
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


for i in range(
    min(500, len(X_train))
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
    "[PASS] Historical sensor availability "
    "constraint verified."
)


# ============================================================
# TRAIN NH4 ESTIMATOR
# ============================================================

print()
print(
    "[INFO] Training causal NH4 estimator..."
)


nh4_estimator = MLPRegressor(
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


nh4_estimator.fit(
    X_train,
    y_train_nh4
)


print(
    "[PASS] Causal NH4 estimator trained."
)


# ============================================================
# TRAIN SNO ESTIMATOR
# ============================================================

print(
    "[INFO] Training causal SNO estimator..."
)


sno_estimator = MLPRegressor(
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


sno_estimator.fit(
    X_train,
    y_train_sno
)


print(
    "[PASS] Causal SNO estimator trained."
)


# ============================================================
# HELD-OUT EVALUATION
# ============================================================

print()
print(
    "[INFO] Evaluating causal estimators "
    "on held-out dynamic activation histories..."
)


evaluation_times = np.asarray(
    eval_indices,
    dtype=np.int64
)

evaluation_times = (
    evaluation_times[
        evaluation_times
        >= HISTORY
    ]
)


X_eval = []

y_eval = []


for t in evaluation_times:

    # --------------------------------------------------------
    # Generate a fresh causal activation history.
    # --------------------------------------------------------

    historical_actions = np.random.randint(
        0,
        ACTION_DIM,
        size=HISTORY
    )

    current_action = np.random.randint(
        0,
        ACTION_DIM
    )

    x = build_estimator_input(
        int(t),
        historical_actions,
        current_action
    )

    y = build_target(
        int(t)
    )

    X_eval.append(
        x
    )

    y_eval.append(
        y
    )


X_eval = np.asarray(
    X_eval,
    dtype=np.float32
)

y_eval = np.asarray(
    y_eval,
    dtype=np.float32
)


assert X_eval.shape[1] == 120

assert y_eval.shape[1] == 2


pred_nh4 = nh4_estimator.predict(
    X_eval
)

pred_sno = sno_estimator.predict(
    X_eval
)


assert np.isfinite(
    pred_nh4
).all()

assert np.isfinite(
    pred_sno
).all()


# ============================================================
# RMSE
# ============================================================

rmse_nh4 = float(
    np.sqrt(
        np.mean(
            (
                pred_nh4
                -
                y_eval[:, 0]
            ) ** 2
        )
    )
)

rmse_sno = float(
    np.sqrt(
        np.mean(
            (
                pred_sno
                -
                y_eval[:, 1]
            ) ** 2
        )
    )
)


# ============================================================
# ZERO-CHANGE BASELINE
# ============================================================

zero_change_nh4 = float(
    np.sqrt(
        np.mean(
            y_eval[:, 0] ** 2
        )
    )
)

zero_change_sno = float(
    np.sqrt(
        np.mean(
            y_eval[:, 1] ** 2
        )
    )
)


improvement_nh4 = (
    1.0
    -
    rmse_nh4
    /
    zero_change_nh4
) * 100.0


improvement_sno = (
    1.0
    -
    rmse_sno
    /
    zero_change_sno
) * 100.0


# ============================================================
# ACTION DIVERSITY
# ============================================================

unique_eval_actions = len(
    np.unique(
        X_eval[:, 12:24],
        axis=0
    )
)


# ============================================================
# RESULTS
# ============================================================

print()
print(
    f"[INFO] Evaluation examples: "
    f"{len(X_eval)}"
)

print(
    f"[INFO] Unique current activation "
    f"configurations in evaluation: "
    f"{unique_eval_actions}"
)

print()
print(
    f"[RESULT] Causal estimator "
    f"ΔNH4 RMSE: {rmse_nh4:.6f}"
)

print(
    f"[RESULT] Zero-change "
    f"ΔNH4 RMSE: {zero_change_nh4:.6f}"
)

print(
    f"[RESULT] Relative ΔNH4 improvement: "
    f"{improvement_nh4:.2f}%"
)

print()
print(
    f"[RESULT] Causal estimator "
    f"ΔSNO RMSE: {rmse_sno:.6f}"
)

print(
    f"[RESULT] Zero-change "
    f"ΔSNO RMSE: {zero_change_sno:.6f}"
)

print(
    f"[RESULT] Relative ΔSNO improvement: "
    f"{improvement_sno:.2f}%"
)


# ============================================================
# INTERFACE VALIDATION
# ============================================================

assert X_train.shape[1] == 120

assert X_eval.shape[1] == 120

assert np.isfinite(
    rmse_nh4
)

assert np.isfinite(
    rmse_sno
)

assert np.isfinite(
    improvement_nh4
)

assert np.isfinite(
    improvement_sno
)


print()
print(
    "[PASS] 120-D estimator interface verified."
)

print(
    "[PASS] Causal historical availability verified."
)

print(
    "[PASS] All 495 current actions represented "
    "during estimator training."
)

print(
    "[PASS] Held-out evaluation completed."
)

print(
    "[PASS] Estimator predictions are finite."
)

print("=" * 70)
print(
    "STEP 63 COMPLETED"
)
print("=" * 70)
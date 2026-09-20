from itertools import combinations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

import bsm2_python
from bsm2_python.bsm1_ol import BSM1OL


# ============================================================
# STEP 25 — FROZEN ESTIMATOR VALIDATION
# ============================================================

DRY_INFLUENT = str(Path(bsm2_python.__file__).resolve().parent / "data" / "dryinfluent.csv")

N_SENSORS = 12
N_ACTIVE = 4
HISTORY = 4
FORECAST_HORIZON = 2

TRAIN_END = 863
EVAL_START = 863

RANDOM_SEED = 20260916

# Number of representative training times per action.
TRAIN_TIMES_PER_ACTION = 20


# ============================================================
# Candidate sensors
# ============================================================

SENSOR_NAMES = [
    "DO_R1",
    "DO_R2",
    "DO_R3",
    "DO_R4",
    "DO_R5",
    "NH4_R1",
    "NH4_R2",
    "NH4_R3",
    "NH4_R4",
    "NH4_R5",
    "SNO_R3",
    "SNO_R5",
]


# ============================================================
# All feasible actions
# ============================================================

ACTIONS = list(
    combinations(
        range(N_SENSORS),
        N_ACTIVE,
    )
)

assert len(ACTIONS) == 495


def action_to_mask(action_index):

    mask = np.zeros(
        N_SENSORS,
        dtype=np.float32,
    )

    mask[
        list(ACTIONS[action_index])
    ] = 1.0

    assert mask.sum() == 4

    return mask


# ============================================================
# Load BSM1
# ============================================================

print("\nLoading BSM1 influent...")

data = np.loadtxt(
    DRY_INFLUENT,
    delimiter=",",
    skiprows=1,
)

model = BSM1OL(
    data_in=data,
    timestep=None,
    evaltime=np.array(
        [8.98958333, 13.98958333]
    ),
)

print("Running BSM1 simulation...")

model.simulate(plot=False)

print("BSM1 simulation completed.")


# ============================================================
# Candidate sensor matrix
# ============================================================

sensor_matrix = np.column_stack(
    [
        model.y_out1_all[:, 7],
        model.y_out2_all[:, 7],
        model.y_out3_all[:, 7],
        model.y_out4_all[:, 7],
        model.y_out5_all[:, 7],

        model.y_out1_all[:, 9],
        model.y_out2_all[:, 9],
        model.y_out3_all[:, 9],
        model.y_out4_all[:, 9],
        model.y_out5_all[:, 9],

        model.y_out3_all[:, 8],
        model.y_out5_all[:, 8],
    ]
)

effluent_nh4 = model.ys_eff_all[:, 9]
effluent_sno = model.ys_eff_all[:, 8]

n_steps = len(effluent_nh4)

print(
    f"Time steps: {n_steps}"
)

print(
    f"Sensor matrix: "
    f"{sensor_matrix.shape}"
)


# ============================================================
# Training-only sensor normalization
# ============================================================

sensor_mean = sensor_matrix[
    :TRAIN_END
].mean(axis=0)

sensor_std = sensor_matrix[
    :TRAIN_END
].std(axis=0)

sensor_std[
    sensor_std == 0.0
] = 1.0

sensor_norm = (
    sensor_matrix
    - sensor_mean
) / sensor_std


# ============================================================
# Training-only target normalization
# ============================================================

delta_nh4 = (
    effluent_nh4[
        FORECAST_HORIZON:
    ]
    -
    effluent_nh4[
        :-FORECAST_HORIZON
    ]
)

delta_sno = (
    effluent_sno[
        FORECAST_HORIZON:
    ]
    -
    effluent_sno[
        :-FORECAST_HORIZON
    ]
)

target_train_end = (
    TRAIN_END
    - FORECAST_HORIZON
)

sigma_nh4 = np.std(
    delta_nh4[
        :target_train_end
    ]
)

sigma_sno = np.std(
    delta_sno[
        :target_train_end
    ]
)

if sigma_nh4 <= 0:
    raise RuntimeError(
        "Invalid NH4 target scale."
    )

if sigma_sno <= 0:
    raise RuntimeError(
        "Invalid SNO target scale."
    )


print("\nTarget scales:")

print(
    f"  sigma ΔNH4: "
    f"{sigma_nh4:.6f}"
)

print(
    f"  sigma ΔSNO: "
    f"{sigma_sno:.6f}"
)


# ============================================================
# Construct causal estimator input
# ============================================================

def build_estimator_input(
    t,
    action_index,
    history_masks,
):

    if t < HISTORY:
        raise ValueError(
            "Insufficient history."
        )

    current_mask = action_to_mask(
        action_index
    )

    historical = []

    for h in range(
        t - HISTORY,
        t,
    ):

        historical_mask = (
            history_masks[h]
        )

        historical_measurement = (
            sensor_norm[h]
            * historical_mask
        )

        historical.extend(
            historical_measurement
        )

        historical.extend(
            historical_mask
        )

    historical = np.asarray(
        historical,
        dtype=np.float32,
    )

    current_measurement = (
        sensor_norm[t]
        * current_mask
    )

    current_measurement = np.asarray(
        current_measurement,
        dtype=np.float32,
    )

    estimator_input = np.concatenate(
        [
            historical,
            current_measurement,
            current_mask,
        ]
    )

    assert estimator_input.shape == (
        120,
    )

    return estimator_input


# ============================================================
# Build training data
#
# Every one of the 495 actions is represented.
#
# The historical activation sequence is generated
# independently for the training examples.
# ============================================================

print(
    "\nBuilding training data..."
)

rng = np.random.default_rng(
    RANDOM_SEED
)

training_times = np.linspace(
    HISTORY,
    TRAIN_END - FORECAST_HORIZON - 1,
    TRAIN_TIMES_PER_ACTION,
    dtype=int,
)

training_times = np.unique(
    training_times
)

print(
    f"Training times per action: "
    f"{len(training_times)}"
)

X_train = []
y_train_nh4 = []
y_train_sno = []


for action_index in range(
    len(ACTIONS)
):

    # --------------------------------------------------------
    # Generate a dynamic historical activation sequence.
    # --------------------------------------------------------

    history_actions = rng.integers(
        0,
        len(ACTIONS),
        size=n_steps,
    )

    history_masks = np.vstack(
        [
            action_to_mask(a)
            for a in history_actions
        ]
    )

    # --------------------------------------------------------
    # Add representative times for this action.
    # --------------------------------------------------------

    for t in training_times:

        X = build_estimator_input(
            t,
            action_index,
            history_masks,
        )

        true_nh4 = (
            effluent_nh4[
                t + FORECAST_HORIZON
            ]
            -
            effluent_nh4[t]
        )

        true_sno = (
            effluent_sno[
                t + FORECAST_HORIZON
            ]
            -
            effluent_sno[t]
        )

        X_train.append(X)

        y_train_nh4.append(
            true_nh4
        )

        y_train_sno.append(
            true_sno
        )


X_train = np.asarray(
    X_train,
    dtype=np.float32,
)

y_train_nh4 = np.asarray(
    y_train_nh4,
    dtype=np.float32,
)

y_train_sno = np.asarray(
    y_train_sno,
    dtype=np.float32,
)


expected_training_examples = (
    len(ACTIONS)
    * len(training_times)
)

assert len(X_train) == (
    expected_training_examples
)

assert X_train.shape[1] == 120


print(
    f"Training examples: "
    f"{len(X_train)}"
)

print(
    f"Expected examples: "
    f"{expected_training_examples}"
)

print(
    f"Training matrix: "
    f"{X_train.shape}"
)


# ============================================================
# Train estimator
# ============================================================

print(
    "\nTraining frozen NH4-N estimator..."
)

nh4_estimator = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=RANDOM_SEED,
)

nh4_estimator.fit(
    X_train,
    y_train_nh4,
)


print(
    "Training frozen SNO estimator..."
)

sno_estimator = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=RANDOM_SEED + 1,
)

sno_estimator.fit(
    X_train,
    y_train_sno,
)


# ============================================================
# Build a held-out dynamic evaluation trajectory
# ============================================================

print(
    "\nBuilding held-out dynamic evaluation..."
)

eval_rng = np.random.default_rng(
    RANDOM_SEED + 100
)

eval_actions = eval_rng.integers(
    0,
    len(ACTIONS),
    size=n_steps,
)

eval_masks = np.vstack(
    [
        action_to_mask(a)
        for a in eval_actions
    ]
)


# ============================================================
# Evaluate estimator on held-out dynamic trajectory
# ============================================================

X_eval = []
y_eval_nh4 = []
y_eval_sno = []

eval_action_indices = []

for t in range(
    EVAL_START,
    n_steps - FORECAST_HORIZON,
):

    action_index = int(
        eval_actions[t]
    )

    X = build_estimator_input(
        t,
        action_index,
        eval_masks,
    )

    true_nh4 = (
        effluent_nh4[
            t + FORECAST_HORIZON
        ]
        -
        effluent_nh4[t]
    )

    true_sno = (
        effluent_sno[
            t + FORECAST_HORIZON
        ]
        -
        effluent_sno[t]
    )

    X_eval.append(X)

    y_eval_nh4.append(
        true_nh4
    )

    y_eval_sno.append(
        true_sno
    )

    eval_action_indices.append(
        action_index
    )


X_eval = np.asarray(
    X_eval,
    dtype=np.float32,
)

y_eval_nh4 = np.asarray(
    y_eval_nh4,
    dtype=np.float32,
)

y_eval_sno = np.asarray(
    y_eval_sno,
    dtype=np.float32,
)

pred_nh4 = nh4_estimator.predict(
    X_eval
)

pred_sno = sno_estimator.predict(
    X_eval
)


# ============================================================
# Metrics
# ============================================================

rmse_nh4 = np.sqrt(
    mean_squared_error(
        y_eval_nh4,
        pred_nh4,
    )
)

rmse_sno = np.sqrt(
    mean_squared_error(
        y_eval_sno,
        pred_sno,
    )
)

baseline_nh4 = np.sqrt(
    mean_squared_error(
        y_eval_nh4,
        np.zeros_like(
            y_eval_nh4
        ),
    )
)

baseline_sno = np.sqrt(
    mean_squared_error(
        y_eval_sno,
        np.zeros_like(
            y_eval_sno
        ),
    )
)

improvement_nh4 = (
    100.0
    * (
        baseline_nh4
        - rmse_nh4
    )
    / baseline_nh4
)

improvement_sno = (
    100.0
    * (
        baseline_sno
        - rmse_sno
    )
    / baseline_sno
)


# ============================================================
# Action coverage
# ============================================================

train_action_coverage = len(
    ACTIONS
)

eval_action_coverage = len(
    np.unique(
        eval_action_indices
    )
)


# ============================================================
# Frozen prediction test
#
# Calling the estimator twice with identical input must
# return identical predictions.
# ============================================================

test_input = X_eval[
    :10
]

prediction_a = nh4_estimator.predict(
    test_input
)

prediction_b = nh4_estimator.predict(
    test_input
)

assert np.allclose(
    prediction_a,
    prediction_b,
)


# ============================================================
# Reward compatibility test
# ============================================================

def calculate_reward(
    predicted_nh4,
    true_nh4,
    predicted_sno,
    true_sno,
):

    error_nh4 = (
        predicted_nh4
        - true_nh4
    )

    error_sno = (
        predicted_sno
        - true_sno
    )

    normalized_nh4 = (
        error_nh4 ** 2
        /
        sigma_nh4 ** 2
    )

    normalized_sno = (
        error_sno ** 2
        /
        sigma_sno ** 2
    )

    return -(
        0.5 * normalized_nh4
        +
        0.5 * normalized_sno
    )


rewards = np.array(
    [
        calculate_reward(
            pred_nh4[i],
            y_eval_nh4[i],
            pred_sno[i],
            y_eval_sno[i],
        )
        for i in range(
            len(X_eval)
        )
    ]
)

assert np.all(
    np.isfinite(rewards)
)


# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 70)
print("STEP 25 — FROZEN ESTIMATOR VALIDATION")
print("=" * 70)

print("\nEstimator:")
print("  Architecture          : 120 → 128 → 64 → 1")
print("  Separate NH4/SNO models")
print("  Estimator is frozen after training.")

print("\nTraining:")
print(
    f"  Actions represented   : "
    f"{train_action_coverage}/495"
)

print(
    f"  Training examples     : "
    f"{len(X_train)}"
)

print(
    f"  Input dimension       : "
    f"{X_train.shape[1]}"
)

print("\nHeld-out dynamic evaluation:")
print(
    f"  Evaluation examples   : "
    f"{len(X_eval)}"
)

print(
    f"  Unique actions        : "
    f"{eval_action_coverage}/495"
)

print("\nΔNH4-N:")
print(
    f"  Zero-change RMSE      : "
    f"{baseline_nh4:.6f}"
)

print(
    f"  Frozen estimator RMSE : "
    f"{rmse_nh4:.6f}"
)

print(
    f"  Relative improvement : "
    f"{improvement_nh4:.2f}%"
)

print("\nΔSNO:")
print(
    f"  Zero-change RMSE      : "
    f"{baseline_sno:.6f}"
)

print(
    f"  Frozen estimator RMSE : "
    f"{rmse_sno:.6f}"
)

print(
    f"  Relative improvement : "
    f"{improvement_sno:.2f}%"
)

print("\nReward:")
print(
    f"  Minimum reward       : "
    f"{rewards.min():.6f}"
)

print(
    f"  Maximum reward       : "
    f"{rewards.max():.6f}"
)

print(
    f"  Mean reward          : "
    f"{rewards.mean():.6f}"
)

print("\nFrozen-estimator test:")
print(
    "  [PASS] Repeated predictions "
    "are identical."
)

print(
    "  [PASS] Reward values are finite."
)

print("\n" + "=" * 70)
print("STEP 25 COMPLETED")
print("=" * 70)
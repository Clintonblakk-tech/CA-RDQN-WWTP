import itertools
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import bsm2_python
from bsm2_python.bsm1_ol import BSM1OL


# ============================================================
# CONFIGURATION
# ============================================================

DRY_INFLUENT = str(Path(bsm2_python.__file__).resolve().parent / "data" / "dryinfluent.csv")

HISTORY = 4
N_SENSORS = 12
ACTIVE_SENSORS = 4
FORECAST_STEPS = 2

TRAIN_END = 863

RANDOM_SEED = 42


# ============================================================
# SENSOR DEFINITIONS
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


# BSM1 state indices
DO_INDEX = 7
SNO_INDEX = 8
NH4_INDEX = 9


# ============================================================
# ACTION SPACE
# ============================================================

ACTIONS = list(
    itertools.combinations(
        range(N_SENSORS),
        ACTIVE_SENSORS
    )
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SHARED ESTIMATOR BASELINE EXPERIMENT")
print("=" * 70)

print(
    f"\nCandidate sensors: {N_SENSORS}"
)

print(
    f"Feasible actions: {len(ACTIONS)}"
)

print(
    f"History length: {HISTORY}"
)

print(
    f"Forecast horizon: {FORECAST_STEPS} × 15 minutes"
)


# ============================================================
# LOAD BSM1 DATA
# ============================================================

print("\nLoading BSM1 dry influent data...")

data = np.loadtxt(
    DRY_INFLUENT,
    delimiter=",",
    skiprows=1
)

print(
    f"Influent data shape: {data.shape}"
)


# ============================================================
# RUN BSM1
# ============================================================

print("\nRunning BSM1 simulation...")

model = BSM1OL(
    data_in=data,
    timestep=None,
    evaltime=np.array(
        [8.98958333, 13.98958333]
    ),
)

model.simulate(
    plot=False
)

print("BSM1 simulation complete.")


# ============================================================
# EXTRACT PROCESS SENSORS
# ============================================================

sensor_matrix = np.column_stack(
    [
        model.y_out1_all[:, DO_INDEX],
        model.y_out2_all[:, DO_INDEX],
        model.y_out3_all[:, DO_INDEX],
        model.y_out4_all[:, DO_INDEX],
        model.y_out5_all[:, DO_INDEX],

        model.y_out1_all[:, NH4_INDEX],
        model.y_out2_all[:, NH4_INDEX],
        model.y_out3_all[:, NH4_INDEX],
        model.y_out4_all[:, NH4_INDEX],
        model.y_out5_all[:, NH4_INDEX],

        model.y_out3_all[:, SNO_INDEX],
        model.y_out5_all[:, SNO_INDEX],
    ]
)


# ============================================================
# TARGET SERIES
# ============================================================

effluent_nh4 = model.ys_eff_all[:, NH4_INDEX]
effluent_sno = model.ys_eff_all[:, SNO_INDEX]

n_steps = sensor_matrix.shape[0]


print(
    f"\nSensor matrix shape: {sensor_matrix.shape}"
)


# ============================================================
# RANDOM SENSOR-ACTIVATION POLICY
#
# This is the NON-RL sensing baseline.
#
# Exactly four sensors are active at every time step.
#
# The random sequence is generated once using a fixed seed
# so that the experiment is reproducible.
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

random_action_ids = rng.integers(
    low=0,
    high=len(ACTIONS),
    size=n_steps
)


# ============================================================
# ACTIVATION HISTORY
# ============================================================

activation_history = np.zeros(
    (n_steps, N_SENSORS),
    dtype=int
)

for t in range(n_steps):

    action_id = random_action_ids[t]

    selected = ACTIONS[action_id]

    activation_history[
        t,
        list(selected)
    ] = 1


# ============================================================
# VERIFY FOUR-SENSOR CONSTRAINT
# ============================================================

active_counts = activation_history.sum(
    axis=1
)

if not np.all(
    active_counts == ACTIVE_SENSORS
):

    raise RuntimeError(
        "Random baseline violated the four-sensor constraint."
    )

print(
    "Random 4-of-12 activation policy: VERIFIED"
)


# ============================================================
# BUILD CAUSAL STATE
# ============================================================

def build_state(time_index):
    """
    Construct the causal estimator state.

    For each of the previous four time steps:

        12 measurement values
        +
        12 availability indicators

    Total:

        4 × 24 = 96 features

    Inactive sensors have measurement value zero, while
    the availability mask identifies that they were not
    observed.
    """

    if time_index < HISTORY:

        raise ValueError(
            "Insufficient history."
        )

    start_index = (
        time_index - HISTORY
    )

    measurements = np.zeros(
        (HISTORY, N_SENSORS),
        dtype=float
    )

    masks = np.zeros(
        (HISTORY, N_SENSORS),
        dtype=float
    )

    for h in range(HISTORY):

        source_index = (
            start_index + h
        )

        masks[h] = activation_history[
            source_index
        ]

        active = (
            activation_history[
                source_index
            ] == 1
        )

        measurements[
            h,
            active
        ] = sensor_matrix[
            source_index,
            active
        ]

    state = np.concatenate(
        [
            measurements,
            masks
        ],
        axis=1
    )

    return state.flatten()


# ============================================================
# BUILD SUPERVISED DATASET
# ============================================================

X = []
Y = []


for t in range(
    HISTORY,
    n_steps - FORECAST_STEPS
):

    state = build_state(t)

    delta_nh4 = (
        effluent_nh4[
            t + FORECAST_STEPS
        ]
        -
        effluent_nh4[t]
    )

    delta_sno = (
        effluent_sno[
            t + FORECAST_STEPS
        ]
        -
        effluent_sno[t]
    )

    X.append(state)

    Y.append(
        [
            delta_nh4,
            delta_sno
        ]
    )


X = np.asarray(
    X,
    dtype=float
)

Y = np.asarray(
    Y,
    dtype=float
)


# ============================================================
# TRAIN / EVALUATION SPLIT
#
# TRAIN_END is expressed in the original BSM1 time index.
# ============================================================

example_indices = np.arange(
    HISTORY,
    n_steps - FORECAST_STEPS
)

train_mask = (
    example_indices < TRAIN_END
)

eval_mask = (
    example_indices >= TRAIN_END
)


X_train = X[
    train_mask
]

X_eval = X[
    eval_mask
]

Y_train = Y[
    train_mask
]

Y_eval = Y[
    eval_mask
]


print("\nDataset:")
print(
    f"Total examples: {len(X)}"
)

print(
    f"Training examples: {len(X_train)}"
)

print(
    f"Evaluation examples: {len(X_eval)}"
)

print(
    f"State dimension: {X.shape[1]}"
)


# ============================================================
# STANDARDIZE INPUTS
#
# Training data only is used to fit the scaler.
# ============================================================

x_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(
    X_train
)

X_eval_scaled = x_scaler.transform(
    X_eval
)


# ============================================================
# SHARED ESTIMATOR
# ============================================================

print(
    "\nTraining shared estimator..."
)

estimator = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=64,
    learning_rate_init=1e-3,
    max_iter=500,
    random_state=RANDOM_SEED,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
)

estimator.fit(
    X_train_scaled,
    Y_train
)

print(
    "Shared estimator training complete."
)


# ============================================================
# SHARED ESTIMATOR PREDICTIONS
# ============================================================

Y_pred = estimator.predict(
    X_eval_scaled
)


# ============================================================
# RANDOM 4-OF-12 BASELINE RMSE
# ============================================================

random_nh4_rmse = np.sqrt(
    mean_squared_error(
        Y_eval[:, 0],
        Y_pred[:, 0]
    )
)

random_sno_rmse = np.sqrt(
    mean_squared_error(
        Y_eval[:, 1],
        Y_pred[:, 1]
    )
)


# ============================================================
# ZERO-CHANGE BASELINE
#
# Prediction:
#
#     ΔY = 0
#
# ============================================================

zero_prediction = np.zeros_like(
    Y_eval
)

zero_nh4_rmse = np.sqrt(
    mean_squared_error(
        Y_eval[:, 0],
        zero_prediction[:, 0]
    )
)

zero_sno_rmse = np.sqrt(
    mean_squared_error(
        Y_eval[:, 1],
        zero_prediction[:, 1]
    )
)


# ============================================================
# PERSISTENCE-CHANGE BASELINE
#
# For a change target:
#
#     Y(t+2) - Y(t)
#
# persistence predicts:
#
#     Y(t+2) - Y(t) = 0
#
# Therefore this is numerically identical to the
# zero-change baseline.
#
# It is reported separately for clarity.
# ============================================================

persistence_nh4_rmse = zero_nh4_rmse
persistence_sno_rmse = zero_sno_rmse


# ============================================================
# RELATIVE CHANGE FROM ZERO-CHANGE BASELINE
# ============================================================

nh4_improvement = (
    100.0
    *
    (
        zero_nh4_rmse
        -
        random_nh4_rmse
    )
    /
    zero_nh4_rmse
)

sno_improvement = (
    100.0
    *
    (
        zero_sno_rmse
        -
        random_sno_rmse
    )
    /
    zero_sno_rmse
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BASELINE RESULTS")
print("=" * 70)

print(
    "\nEvaluation period:"
)

print(
    f"  BSM1 indices {TRAIN_END}–{n_steps - 1}"
)

print(
    f"  Evaluation observations: {len(Y_eval)}"
)


print("\nΔNH4-N RMSE:")

print(
    f"  Zero-change baseline: "
    f"{zero_nh4_rmse:.6f}"
)

print(
    f"  Persistence baseline: "
    f"{persistence_nh4_rmse:.6f}"
)

print(
    f"  Random 4-of-12 shared estimator: "
    f"{random_nh4_rmse:.6f}"
)

print(
    f"  Relative change vs zero-change: "
    f"{nh4_improvement:.2f}%"
)


print("\nΔSNO RMSE:")

print(
    f"  Zero-change baseline: "
    f"{zero_sno_rmse:.6f}"
)

print(
    f"  Persistence baseline: "
    f"{persistence_sno_rmse:.6f}"
)

print(
    f"  Random 4-of-12 shared estimator: "
    f"{random_sno_rmse:.6f}"
)

print(
    f"  Relative change vs zero-change: "
    f"{sno_improvement:.2f}%"
)


# ============================================================
# ACTION COVERAGE
# ============================================================

evaluation_action_ids = (
    random_action_ids[
        TRAIN_END:
        n_steps - FORECAST_STEPS
    ]
)

unique_eval_actions = np.unique(
    evaluation_action_ids
)

print(
    "\nRandom-policy action coverage:"
)

print(
    f"  Unique actions observed: "
    f"{len(unique_eval_actions)}"
)

print(
    f"  Possible actions: "
    f"{len(ACTIONS)}"
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("BASELINE VALIDATION")
print("=" * 70)

checks_passed = True


if X.shape[1] == 96:

    print(
        "PASSED: state dimension = 96"
    )

else:

    print(
        "FAILED: state dimension is not 96"
    )

    checks_passed = False


if len(ACTIONS) == 495:

    print(
        "PASSED: feasible action count = 495"
    )

else:

    print(
        "FAILED: feasible action count is not 495"
    )

    checks_passed = False


if np.all(
    active_counts == 4
):

    print(
        "PASSED: random policy activates exactly "
        "four sensors at every time step"
    )

else:

    print(
        "FAILED: four-sensor constraint violated"
    )

    checks_passed = False


if np.all(
    np.isfinite(Y_pred)
):

    print(
        "PASSED: shared estimator predictions are finite"
    )

else:

    print(
        "FAILED: non-finite estimator predictions"
    )

    checks_passed = False


if len(X_train) > 0 and len(X_eval) > 0:

    print(
        "PASSED: training/evaluation split is valid"
    )

else:

    print(
        "FAILED: invalid training/evaluation split"
    )

    checks_passed = False


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)

if checks_passed:

    print(
        "SHARED ESTIMATOR BASELINE TEST PASSED"
    )

    print("=" * 70)

    print(
        "\nThe random 4-of-12 policy now provides a "
        "non-RL reference for evaluating the future "
        "constraint-aware DQN."
    )

else:

    print(
        "SHARED ESTIMATOR BASELINE TEST FAILED"
    )

    print("=" * 70)
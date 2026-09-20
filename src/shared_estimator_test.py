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
EVAL_START = 863


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
# LOAD BSM1
# ============================================================

print("=" * 70)
print("SHARED ESTIMATOR TEST")
print("=" * 70)

print("\nLoading BSM1 dry influent data...")

data = np.loadtxt(
    DRY_INFLUENT,
    delimiter=",",
    skiprows=1
)

print(
    f"Influent data shape: {data.shape}"
)

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
# EXTRACT 12 PROCESS SENSORS
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

print(
    f"Number of feasible actions: {len(ACTIONS)}"
)


# ============================================================
# BUILD TRAINING EXAMPLES
#
# IMPORTANT:
#
# The historical activation pattern is deliberately generated
# first.
#
# The estimator receives only measurements that were actually
# acquired.
#
# ============================================================

rng = np.random.default_rng(42)

activation_history = np.zeros(
    (n_steps, N_SENSORS),
    dtype=int
)


# Random feasible action at every decision time.
# This is only for generating a diverse training dataset.
for t in range(n_steps):

    action_id = rng.integers(
        0,
        len(ACTIONS)
    )

    selected = ACTIONS[action_id]

    activation_history[
        t,
        list(selected)
    ] = 1


# ============================================================
# BUILD CAUSAL STATE
# ============================================================

def build_state(
    time_index
):
    """
    Build a causal 96-dimensional state.

    Four historical time steps are used.

    For each historical step:
        12 masked measurements
        +
        12 availability indicators

    Total:
        4 × (12 + 12) = 96
    """

    if time_index < HISTORY:

        raise ValueError(
            "Insufficient history."
        )

    start = time_index - HISTORY

    measurements = np.zeros(
        (HISTORY, N_SENSORS),
        dtype=float
    )

    masks = np.zeros(
        (HISTORY, N_SENSORS),
        dtype=float
    )

    for h in range(HISTORY):

        source_index = start + h

        masks[
            h
        ] = activation_history[
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
# BUILD DATASET
# ============================================================

X = []
Y = []
ACTION_IDS = []


for t in range(
    HISTORY,
    n_steps - FORECAST_STEPS
):

    state = build_state(t)

    # --------------------------------------------------------
    # Current action
    #
    # The action at time t is part of the state because its
    # activation mask is the current scheduler configuration.
    # --------------------------------------------------------

    current_action = activation_history[t]

    # --------------------------------------------------------
    # Target = 30-minute change
    # --------------------------------------------------------

    delta_nh4 = (
        effluent_nh4[t + FORECAST_STEPS]
        -
        effluent_nh4[t]
    )

    delta_sno = (
        effluent_sno[t + FORECAST_STEPS]
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

    # Recover action ID from the active sensors.
    selected = tuple(
        np.where(
            current_action == 1
        )[0]
    )

    action_id = ACTIONS.index(
        selected
    )

    ACTION_IDS.append(action_id)


X = np.asarray(
    X,
    dtype=float
)

Y = np.asarray(
    Y,
    dtype=float
)

ACTION_IDS = np.asarray(
    ACTION_IDS,
    dtype=int
)


# ============================================================
# TRAIN / EVALUATION SPLIT
# ============================================================

train_mask = (
    np.arange(len(X))
    <
    TRAIN_END - HISTORY
)

eval_mask = ~train_mask


X_train = X[train_mask]
X_eval = X[eval_mask]

Y_train = Y[train_mask]
Y_eval = Y[eval_mask]

action_eval = ACTION_IDS[eval_mask]


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
# STANDARDIZATION
#
# Fit scaling ONLY on the training data.
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

print("\nTraining shared estimator...")

estimator = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=64,
    learning_rate_init=1e-3,
    max_iter=500,
    random_state=42,
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
# PREDICTION
# ============================================================

Y_pred = estimator.predict(
    X_eval_scaled
)


# ============================================================
# METRICS
# ============================================================

nh4_rmse = np.sqrt(
    mean_squared_error(
        Y_eval[:, 0],
        Y_pred[:, 0]
    )
)

sno_rmse = np.sqrt(
    mean_squared_error(
        Y_eval[:, 1],
        Y_pred[:, 1]
    )
)


print("\n" + "=" * 70)
print("SHARED ESTIMATOR RESULTS")
print("=" * 70)

print(
    f"\nΔNH4-N RMSE: {nh4_rmse:.6f}"
)

print(
    f"ΔSNO RMSE:   {sno_rmse:.6f}"
)


# ============================================================
# ACTION COVERAGE
# ============================================================

unique_actions = np.unique(
    action_eval
)

print(
    f"\nUnique actions observed during evaluation: "
    f"{len(unique_actions)}"
)

print(
    f"Possible actions: {len(ACTIONS)}"
)


# ============================================================
# FINAL CHECKS
# ============================================================

print("\n" + "=" * 70)
print("SHARED ESTIMATOR VALIDATION")
print("=" * 70)


checks_passed = True


if X.shape[1] != 96:

    print(
        "FAILED: state dimension is not 96."
    )

    checks_passed = False

else:

    print(
        "PASSED: 96-dimensional causal state."
    )


if len(ACTIONS) != 495:

    print(
        "FAILED: action space is not 495."
    )

    checks_passed = False

else:

    print(
        "PASSED: 495 feasible actions."
    )


if Y.shape[1] != 2:

    print(
        "FAILED: estimator does not have two targets."
    )

    checks_passed = False

else:

    print(
        "PASSED: two prediction targets."
    )


if np.all(np.isfinite(Y_pred)):

    print(
        "PASSED: estimator produced finite predictions."
    )

else:

    print(
        "FAILED: non-finite predictions detected."
    )

    checks_passed = False


if checks_passed:

    print("\n" + "=" * 70)
    print("SHARED ESTIMATOR TEST PASSED")
    print("=" * 70)

    print(
        "\nA single estimator can now be evaluated using "
        "the corrected causal sensor interface."
    )

else:

    print("\n" + "=" * 70)
    print("SHARED ESTIMATOR TEST FAILED")
    print("=" * 70)
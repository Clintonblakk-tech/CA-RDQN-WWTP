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

TRAIN_TIME_POINTS_PER_ACTION = 40

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


# ============================================================
# BSM1 STATE INDICES
# ============================================================

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
print("SHARED ESTIMATOR ACTION-INFORMATION DIAGNOSTIC")
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
    f"Forecast horizon: "
    f"{FORECAST_STEPS} × 15 minutes"
)

print(
    f"Training time points per action: "
    f"{TRAIN_TIME_POINTS_PER_ACTION}"
)


# ============================================================
# VERIFY ACTION SPACE
# ============================================================

if len(ACTIONS) != 495:

    raise RuntimeError(
        "Expected 495 feasible 4-of-12 actions."
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

effluent_nh4 = model.ys_eff_all[
    :,
    NH4_INDEX
]

effluent_sno = model.ys_eff_all[
    :,
    SNO_INDEX
]

n_steps = sensor_matrix.shape[0]


print(
    f"\nSensor matrix shape: "
    f"{sensor_matrix.shape}"
)


# ============================================================
# STATIC ACTION ACTIVATION
# ============================================================

def create_static_activation_history(action_id):
    """
    Create a static 4-of-12 sensing configuration.

    The same four sensors are active throughout the entire
    sequence.

    This is used ONLY for the action-information diagnostic.
    """

    activation = np.zeros(
        (n_steps, N_SENSORS),
        dtype=int
    )

    selected = ACTIONS[action_id]

    activation[
        :,
        list(selected)
    ] = 1

    return activation


# ============================================================
# BUILD CAUSAL STATE
# ============================================================

def build_state(
    time_index,
    activation_history
):
    """
    Construct the 96-dimensional causal state.

    Four historical time steps are represented.

    Each time step contains:

        12 masked measurements
        +
        12 availability indicators

    Therefore:

        4 × 24 = 96 features

    No future target information is included.
    """

    if time_index < HISTORY:

        raise ValueError(
            "Insufficient historical data."
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
# BUILD TARGET
# ============================================================

def build_target(time_index):
    """
    Construct the 30-minute change target.

        ΔNH4-N = NH4(t+2) - NH4(t)

        ΔSNO   = SNO(t+2) - SNO(t)

    BSM1 is sampled at 15-minute intervals.
    Therefore two steps correspond to 30 minutes.
    """

    delta_nh4 = (
        effluent_nh4[
            time_index + FORECAST_STEPS
        ]
        -
        effluent_nh4[
            time_index
        ]
    )

    delta_sno = (
        effluent_sno[
            time_index + FORECAST_STEPS
        ]
        -
        effluent_sno[
            time_index
        ]
    )

    return np.array(
        [
            delta_nh4,
            delta_sno
        ],
        dtype=float
    )


# ============================================================
# TRAINING TIME POINTS
#
# Use exactly 40 points distributed across the training
# period.
#
# Every one of the 495 actions will use the SAME temporal
# training points.
# ============================================================

training_time_indices = np.linspace(
    HISTORY,
    TRAIN_END - FORECAST_STEPS - 1,
    TRAIN_TIME_POINTS_PER_ACTION,
    dtype=int
)

print(
    "\nTraining time indices:"
)

print(
    training_time_indices
)


# ============================================================
# BUILD SHARED TRAINING DATASET
#
# ALL 495 ACTIONS ARE REPRESENTED.
#
# 495 actions × 40 time points
# = 19,800 examples
# ============================================================

print(
    "\nConstructing shared-estimator training dataset..."
)

X_train_parts = []
Y_train_parts = []


for action_id in range(
    len(ACTIONS)
):

    activation_history = (
        create_static_activation_history(
            action_id
        )
    )

    for time_index in training_time_indices:

        state = build_state(
            time_index,
            activation_history
        )

        target = build_target(
            time_index
        )

        X_train_parts.append(
            state
        )

        Y_train_parts.append(
            target
        )


X_train = np.asarray(
    X_train_parts,
    dtype=float
)

Y_train = np.asarray(
    Y_train_parts,
    dtype=float
)


expected_training_examples = (
    len(ACTIONS)
    *
    TRAIN_TIME_POINTS_PER_ACTION
)


print(
    f"Training configurations: "
    f"{len(ACTIONS)}"
)

print(
    f"Training examples: "
    f"{len(X_train)}"
)

print(
    f"Expected training examples: "
    f"{expected_training_examples}"
)


# ============================================================
# VERIFY TRAINING DATASET
# ============================================================

if len(X_train) != expected_training_examples:

    raise RuntimeError(
        "Unexpected number of training examples."
    )

if X_train.shape[1] != 96:

    raise RuntimeError(
        "Training state dimension is not 96."
    )

if Y_train.shape[1] != 2:

    raise RuntimeError(
        "Training target dimension is not 2."
    )


# ============================================================
# STANDARDIZE INPUTS
#
# IMPORTANT:
#
# The scaler is fitted ONLY on training data.
# ============================================================

x_scaler = StandardScaler()

X_train_scaled = (
    x_scaler.fit_transform(
        X_train
    )
)


# ============================================================
# TRAIN ONE SHARED ESTIMATOR
# ============================================================

print(
    "\nTraining shared estimator..."
)

estimator = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=300,
    random_state=RANDOM_SEED,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
)

estimator.fit(
    X_train_scaled,
    Y_train
)

print(
    "Shared estimator training complete."
)

print(
    f"Training iterations used: "
    f"{estimator.n_iter_}"
)


# ============================================================
# HELD-OUT EVALUATION
#
# Every one of the 495 actions is evaluated using the SAME
# trained estimator.
# ============================================================

evaluation_time_indices = np.arange(
    TRAIN_END,
    n_steps - FORECAST_STEPS
)

print(
    "\nEvaluation time points: "
    f"{len(evaluation_time_indices)}"
)

print(
    "Evaluating all 495 feasible configurations..."
)


results = []


for action_id, selected in enumerate(
    ACTIONS
):

    activation_history = (
        create_static_activation_history(
            action_id
        )
    )

    X_eval = []
    Y_eval = []

    for time_index in evaluation_time_indices:

        state = build_state(
            time_index,
            activation_history
        )

        target = build_target(
            time_index
        )

        X_eval.append(
            state
        )

        Y_eval.append(
            target
        )

    X_eval = np.asarray(
        X_eval,
        dtype=float
    )

    Y_eval = np.asarray(
        Y_eval,
        dtype=float
    )

    X_eval_scaled = (
        x_scaler.transform(
            X_eval
        )
    )

    Y_pred = estimator.predict(
        X_eval_scaled
    )

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

    combined_rmse = np.sqrt(
        (
            nh4_rmse ** 2
            +
            sno_rmse ** 2
        )
        /
        2.0
    )

    results.append(
        (
            action_id,
            selected,
            nh4_rmse,
            sno_rmse,
            combined_rmse
        )
    )


# ============================================================
# SORT CONFIGURATIONS
# ============================================================

results = sorted(
    results,
    key=lambda result: result[4]
)


# ============================================================
# EXTRACT ERROR DISTRIBUTIONS
# ============================================================

nh4_values = np.asarray(
    [
        result[2]
        for result in results
    ],
    dtype=float
)

sno_values = np.asarray(
    [
        result[3]
        for result in results
    ],
    dtype=float
)

combined_values = np.asarray(
    [
        result[4]
        for result in results
    ],
    dtype=float
)


# ============================================================
# ZERO-CHANGE BASELINE
# ============================================================

Y_eval_all = np.asarray(
    [
        build_target(time_index)
        for time_index in evaluation_time_indices
    ],
    dtype=float
)


zero_nh4_rmse = np.sqrt(
    mean_squared_error(
        Y_eval_all[:, 0],
        np.zeros(
            len(Y_eval_all)
        )
    )
)

zero_sno_rmse = np.sqrt(
    mean_squared_error(
        Y_eval_all[:, 1],
        np.zeros(
            len(Y_eval_all)
        )
    )
)


# ============================================================
# ACTION VARIABILITY
# ============================================================

nh4_range = (
    nh4_values.max()
    -
    nh4_values.min()
)

sno_range = (
    sno_values.max()
    -
    sno_values.min()
)

combined_range = (
    combined_values.max()
    -
    combined_values.min()
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ACTION-INFORMATION DIAGNOSTIC RESULTS")
print("=" * 70)


print(
    "\nTraining:"
)

print(
    f"  Actions represented: "
    f"{len(ACTIONS)}"
)

print(
    f"  Time points per action: "
    f"{TRAIN_TIME_POINTS_PER_ACTION}"
)

print(
    f"  Total training examples: "
    f"{len(X_train)}"
)


print(
    "\nEvaluation:"
)

print(
    f"  Evaluation time points: "
    f"{len(evaluation_time_indices)}"
)

print(
    f"  Configurations evaluated: "
    f"{len(results)}"
)


# ============================================================
# NH4 RESULTS
# ============================================================

print(
    "\nΔNH4-N RMSE across 495 configurations:"
)

print(
    f"  Mean:   {nh4_values.mean():.6f}"
)

print(
    f"  Median: {np.median(nh4_values):.6f}"
)

print(
    f"  Std:    {nh4_values.std():.6f}"
)

print(
    f"  Min:    {nh4_values.min():.6f}"
)

print(
    f"  Max:    {nh4_values.max():.6f}"
)

print(
    f"  Range:  {nh4_range:.6f}"
)


# ============================================================
# SNO RESULTS
# ============================================================

print(
    "\nΔSNO RMSE across 495 configurations:"
)

print(
    f"  Mean:   {sno_values.mean():.6f}"
)

print(
    f"  Median: {np.median(sno_values):.6f}"
)

print(
    f"  Std:    {sno_values.std():.6f}"
)

print(
    f"  Min:    {sno_values.min():.6f}"
)

print(
    f"  Max:    {sno_values.max():.6f}"
)

print(
    f"  Range:  {sno_range:.6f}"
)


# ============================================================
# COMBINED RESULTS
# ============================================================

print(
    "\nCombined RMSE across 495 configurations:"
)

print(
    f"  Mean:   {combined_values.mean():.6f}"
)

print(
    f"  Median: {np.median(combined_values):.6f}"
)

print(
    f"  Std:    {combined_values.std():.6f}"
)

print(
    f"  Min:    {combined_values.min():.6f}"
)

print(
    f"  Max:    {combined_values.max():.6f}"
)

print(
    f"  Range:  {combined_range:.6f}"
)


# ============================================================
# BEST CONFIGURATIONS
# ============================================================

print("\n" + "=" * 70)
print("LOWEST-ERROR CONFIGURATIONS")
print("=" * 70)


for rank, result in enumerate(
    results[:10],
    start=1
):

    (
        action_id,
        selected,
        nh4_rmse,
        sno_rmse,
        combined
    ) = result

    names = [
        SENSOR_NAMES[index]
        for index in selected
    ]

    print(
        f"\n{rank}. Action {action_id}"
    )

    print(
        f"   Sensors: {names}"
    )

    print(
        f"   ΔNH4-N RMSE: {nh4_rmse:.6f}"
    )

    print(
        f"   ΔSNO RMSE:   {sno_rmse:.6f}"
    )

    print(
        f"   Combined:    {combined:.6f}"
    )


# ============================================================
# WORST CONFIGURATIONS
# ============================================================

print("\n" + "=" * 70)
print("HIGHEST-ERROR CONFIGURATIONS")
print("=" * 70)


for rank, result in enumerate(
    results[-10:][::-1],
    start=1
):

    (
        action_id,
        selected,
        nh4_rmse,
        sno_rmse,
        combined
    ) = result

    names = [
        SENSOR_NAMES[index]
        for index in selected
    ]

    print(
        f"\n{rank}. Action {action_id}"
    )

    print(
        f"   Sensors: {names}"
    )

    print(
        f"   ΔNH4-N RMSE: {nh4_rmse:.6f}"
    )

    print(
        f"   ΔSNO RMSE:   {sno_rmse:.6f}"
    )

    print(
        f"   Combined:    {combined:.6f}"
    )


# ============================================================
# ZERO-CHANGE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("ZERO-CHANGE BASELINE COMPARISON")
print("=" * 70)

print(
    f"\nZero-change ΔNH4-N RMSE: "
    f"{zero_nh4_rmse:.6f}"
)

print(
    f"Mean configuration ΔNH4-N RMSE: "
    f"{nh4_values.mean():.6f}"
)

print(
    f"Best configuration ΔNH4-N RMSE: "
    f"{nh4_values.min():.6f}"
)


print(
    f"\nZero-change ΔSNO RMSE: "
    f"{zero_sno_rmse:.6f}"
)

print(
    f"Mean configuration ΔSNO RMSE: "
    f"{sno_values.mean():.6f}"
)

print(
    f"Best configuration ΔSNO RMSE: "
    f"{sno_values.min():.6f}"
)


# ============================================================
# ACTION VARIABILITY TEST
# ============================================================

print("\n" + "=" * 70)
print("ACTION VARIABILITY TEST")
print("=" * 70)

print(
    f"\nΔNH4-N RMSE range: "
    f"{nh4_range:.6f}"
)

print(
    f"ΔSNO RMSE range: "
    f"{sno_range:.6f}"
)

print(
    f"Combined RMSE range: "
    f"{combined_range:.6f}"
)


if nh4_range > 0.01:

    print(
        "\nΔNH4-N: configuration-dependent "
        "variation detected."
    )

else:

    print(
        "\nΔNH4-N: very small configuration-dependent "
        "variation detected."
    )


if sno_range > 0.01:

    print(
        "ΔSNO: configuration-dependent "
        "variation detected."
    )

else:

    print(
        "ΔSNO: very small configuration-dependent "
        "variation detected."
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("ACTION-INFORMATION DIAGNOSTIC VALIDATION")
print("=" * 70)


checks_passed = True


if len(results) == 495:

    print(
        "PASSED: all 495 feasible configurations evaluated."
    )

else:

    print(
        "FAILED: not all 495 configurations evaluated."
    )

    checks_passed = False


if len(X_train) == 19800:

    print(
        "PASSED: 19,800 training examples constructed."
    )

else:

    print(
        "FAILED: expected 19,800 training examples."
    )

    checks_passed = False


if X_train.shape[1] == 96:

    print(
        "PASSED: training state dimension = 96."
    )

else:

    print(
        "FAILED: training state dimension is not 96."
    )

    checks_passed = False


if Y_train.shape[1] == 2:

    print(
        "PASSED: two prediction targets."
    )

else:

    print(
        "FAILED: target dimension is not 2."
    )

    checks_passed = False


if np.all(
    np.isfinite(
        combined_values
    )
):

    print(
        "PASSED: all configuration errors are finite."
    )

else:

    print(
        "FAILED: non-finite configuration error detected."
    )

    checks_passed = False


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)

if checks_passed:

    print(
        "ACTION-INFORMATION DIAGNOSTIC PASSED"
    )

    print("=" * 70)

    print(
        "\nAll 495 feasible sensor configurations were "
        "represented during shared-estimator training "
        "and subsequently evaluated on the held-out period."
    )

else:

    print(
        "ACTION-INFORMATION DIAGNOSTIC FAILED"
    )

    print("=" * 70)
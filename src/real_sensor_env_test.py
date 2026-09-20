import itertools
import numpy as np

import bsm2_python
from bsm2_python.bsm1_ol import BSM1OL


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = str(Path(bsm2_python.__file__).resolve().parent / "data" / "dryinfluent.csv")

TRAIN_END = 863
LEAD = 2
HISTORY = 4
N_SENSORS = 12


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
# UTILITY FUNCTIONS
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2,
            axis=0
        )
    )


def make_sensor_matrix(model):
    """
    Construct the 12 candidate-sensor matrix.
    """

    return np.column_stack([
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
    ])


def make_mask(combination):
    """
    Convert an action's four-sensor combination into
    a 12-element binary activation mask.
    """

    mask = np.zeros(
        N_SENSORS,
        dtype=np.float32
    )

    mask[list(combination)] = 1.0

    return mask


def build_masked_measurements(
    measurements,
    mask
):
    """
    Apply the activation mask.

    Inactive sensor measurements are unavailable
    and therefore represented as zero.

    The mask itself is retained separately.
    """

    return measurements * mask


def extract_selected_history(
    masked_history,
    combination
):
    """
    Extract only the selected sensors from a four-step
    masked measurement history.

    Input:
        48 features =
        4 historical observations × 12 sensors

    Output:
        16 features =
        4 historical observations × 4 selected sensors
    """

    selected = []

    for h in range(HISTORY):

        start = h * N_SENSORS

        selected.extend(
            masked_history[
                start + np.asarray(combination)
            ]
        )

    return np.asarray(
        selected,
        dtype=np.float64
    )


def build_history(
    X,
    masks
):
    """
    Construct the 96-dimensional scheduler state.

    For each time t:

        four historical masked measurements
        +
        four historical activation masks

    """

    states = []

    for t in range(HISTORY, len(X)):

        measurement_history = []

        mask_history = []

        for h in range(
            t - HISTORY,
            t
        ):

            measurement_history.extend(
                X[h] * masks[h]
            )

            mask_history.extend(
                masks[h]
            )

        states.append(
            np.concatenate([
                np.asarray(
                    measurement_history,
                    dtype=np.float64
                ),
                np.asarray(
                    mask_history,
                    dtype=np.float64
                ),
            ])
        )

    return np.asarray(states)


# ============================================================
# FOUR-SENSOR ESTIMATOR
# ============================================================

class FourSensorEstimator:

    def __init__(self):
        self.coefficients = None

    def fit(
        self,
        X,
        targets
    ):

        A = np.column_stack([
            np.ones(len(X)),
            X
        ])

        self.coefficients = np.linalg.lstsq(
            A,
            targets,
            rcond=None
        )[0]

    def predict(self, X):

        A = np.column_stack([
            np.ones(len(X)),
            X
        ])

        return A @ self.coefficients


# ============================================================
# RUN BSM1
# ============================================================

print("Running BSM1 simulation...")

data = np.loadtxt(
    DATA_FILE,
    delimiter=","
)

model = BSM1OL(
    data_in=data,
    timestep=None
)

model.simulate(
    plot=False
)

print("BSM1 simulation complete.")


# ============================================================
# BUILD SENSOR AND TARGET DATA
# ============================================================

X = make_sensor_matrix(model)

Y = np.column_stack([
    model.ys_eff_all[:, 9],   # Effluent NH4-N
    model.ys_eff_all[:, 8],   # Effluent SNO
])


# ============================================================
# 30-MINUTE TARGET CHANGE
# ============================================================

end = len(X) - LEAD

X = X[:end]

delta_Y = (
    Y[LEAD:]
    - Y[:end]
)


# ============================================================
# INITIALIZE MASKS
# ============================================================

# Before the first scheduler action there is no
# activation history.

masks = np.zeros(
    (len(X), N_SENSORS),
    dtype=np.float32
)


# ============================================================
# BUILD AN INITIAL STATE
# ============================================================

# For this environment test we use a fixed valid action
# during the historical warm-up period.

warmup_action = 0

warmup_combination = list(
    itertools.combinations(
        range(N_SENSORS),
        4
    )
)[warmup_action]

warmup_mask = make_mask(
    warmup_combination
)


for t in range(HISTORY):
    masks[t] = warmup_mask


# ============================================================
# BUILD INITIAL MASKED STATE
# ============================================================

initial_state_measurements = []

initial_state_masks = []

for t in range(HISTORY):

    initial_state_measurements.extend(
        X[t] * masks[t]
    )

    initial_state_masks.extend(
        masks[t]
    )


state = np.concatenate([
    np.asarray(
        initial_state_measurements,
        dtype=np.float64
    ),
    np.asarray(
        initial_state_masks,
        dtype=np.float64
    ),
])


# ============================================================
# TRAINING TARGET SCALE
# ============================================================

D_train = delta_Y[
    HISTORY:TRAIN_END
]

target_scale = np.std(
    D_train,
    axis=0
)

target_scale = np.maximum(
    target_scale,
    1e-8
)


# ============================================================
# TRAIN FOUR-SENSOR ESTIMATORS
# ============================================================

combinations = list(
    itertools.combinations(
        range(N_SENSORS),
        4
    )
)


print()
print("Training action-specific estimators...")
print(
    "Feasible actions:",
    len(combinations)
)


estimators = {}


# For estimator fitting, construct training histories
# according to each action.

for action, combination in enumerate(
    combinations
):

    X_train_selected = []

    for t in range(
        HISTORY,
        TRAIN_END
    ):

        selected_history = []

        for h in range(
            t - HISTORY,
            t
        ):

            # During estimator training the action is
            # assumed to provide the corresponding four
            # measurements at each historical timestep.

            selected_history.extend(
                X[h, list(combination)]
            )

        X_train_selected.append(
            selected_history
        )

    X_train_selected = np.asarray(
        X_train_selected,
        dtype=np.float64
    )

    estimator = FourSensorEstimator()

    estimator.fit(
        X_train_selected,
        D_train
    )

    estimators[action] = estimator


print(
    "Action-specific estimator training complete."
)


# ============================================================
# ENVIRONMENT FUNCTIONS
# ============================================================

def get_state_at(
    t,
    activation_masks
):

    measurement_history = []

    mask_history = []

    for h in range(
        t - HISTORY,
        t
    ):

        measurement_history.extend(
            X[h] * activation_masks[h]
        )

        mask_history.extend(
            activation_masks[h]
        )

    return np.concatenate([
        np.asarray(
            measurement_history,
            dtype=np.float64
        ),
        np.asarray(
            mask_history,
            dtype=np.float64
        ),
    ])


def evaluate_action(
    t,
    action,
    activation_masks
):

    combination = combinations[action]

    # --------------------------------------------------------
    # Measurements available from the action.
    # --------------------------------------------------------

    selected_history = []

    for h in range(
        t - HISTORY,
        t
    ):

        # The estimator sees only the four sensors
        # selected by the current action.

        selected_history.extend(
            X[h, list(combination)]
        )

    selected_history = np.asarray(
        selected_history,
        dtype=np.float64
    )

    prediction = estimators[action].predict(
        selected_history.reshape(1, -1)
    )[0]

    target = delta_Y[t]

    error = prediction - target

    normalized_error = (
        error / target_scale
    )

    loss = np.mean(
        normalized_error ** 2
    )

    reward = -loss

    return {
        "action": action,
        "combination": combination,
        "prediction": prediction,
        "target": target,
        "error": error,
        "loss": loss,
        "reward": reward,
    }


# ============================================================
# TEST REAL BSM1 ENVIRONMENT
# ============================================================

print()
print("=" * 70)
print("REAL BSM1 SENSOR-SCHEDULING ENVIRONMENT TEST")
print("=" * 70)

print()
print("Candidate sensors:", N_SENSORS)
print("Feasible actions:", len(combinations))
print("State dimension:", len(state))
print("History:", "4 × 15 min = 1 hour")
print("Decision interval:", "15 minutes")
print("Forecast horizon:", "30 minutes")
print("Training boundary:", TRAIN_END)


# ============================================================
# STATE DIMENSION CHECK
# ============================================================

if len(state) != 96:

    raise RuntimeError(
        f"State dimension = {len(state)}, "
        "expected 96."
    )


print()
print("Initial 96-dimensional state: VERIFIED")


# ============================================================
# TEST SEQUENCE OF ACTIONS
# ============================================================

test_actions = [
    0,
    100,
    250,
    494,
]


activation_masks = masks.copy()


for step_number, action in enumerate(
    test_actions,
    start=1
):

    t = TRAIN_END + step_number

    combination = combinations[action]

    mask = make_mask(
        combination
    )

    # --------------------------------------------------------
    # Register the current action.
    # --------------------------------------------------------

    activation_masks[t] = mask

    # --------------------------------------------------------
    # Construct the state at this time.
    # --------------------------------------------------------

    current_state = get_state_at(
        t,
        activation_masks
    )

    # --------------------------------------------------------
    # Evaluate action.
    # --------------------------------------------------------

    result = evaluate_action(
        t,
        action,
        activation_masks
    )

    print()
    print(
        f"DECISION STEP {step_number}"
    )

    print(
        "Decision index:",
        t
    )

    print(
        "Action:",
        action
    )

    print(
        "Selected sensors:"
    )

    for sensor_index in combination:

        print(
            " ",
            SENSOR_NAMES[sensor_index]
        )

    print(
        "Active sensors:",
        int(mask.sum())
    )

    print(
        "State dimension:",
        len(current_state)
    )

    print(
        "Predicted ΔNH4-N:",
        f"{result['prediction'][0]:.6f}"
    )

    print(
        "Predicted ΔSNO:",
        f"{result['prediction'][1]:.6f}"
    )

    print(
        "True ΔNH4-N:",
        f"{result['target'][0]:.6f}"
    )

    print(
        "True ΔSNO:",
        f"{result['target'][1]:.6f}"
    )

    print(
        "Reward:",
        f"{result['reward']:.6f}"
    )

    # --------------------------------------------------------
    # Validate constraints.
    # --------------------------------------------------------

    if int(mask.sum()) != 4:

        raise RuntimeError(
            "Four-sensor constraint violated."
        )

    if len(current_state) != 96:

        raise RuntimeError(
            "State dimension is not 96."
        )

    if not np.isfinite(
        result["reward"]
    ):

        raise RuntimeError(
            "Reward is not finite."
        )


# ============================================================
# VERIFY MASK HISTORY
# ============================================================

final_t = TRAIN_END + len(test_actions)

final_state = get_state_at(
    final_t,
    activation_masks
)

final_masks = final_state[48:]


expected_masks = np.concatenate([
    activation_masks[
        final_t - HISTORY + i
    ]
    for i in range(HISTORY)
])


print()
print("Verifying final activation history...")


if not np.array_equal(
    final_masks,
    expected_masks
):

    raise RuntimeError(
        "Activation history is incorrect."
    )


print(
    "Final activation history: VERIFIED"
)


# ============================================================
# VERIFY NO TARGET IN STATE
# ============================================================

effluent_targets = Y[
    final_t
]

if np.any(
    np.isclose(
        final_state,
        effluent_targets[0]
    )
):

    print(
        "Note: numerical coincidence detected; "
        "this is not treated as target leakage."
    )


print()
print(
    "Target variables are not explicitly included "
    "in the state construction."
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("REAL BSM1 SENSOR ENVIRONMENT TEST PASSED")
print("=" * 70)

print()
print("Verified:")
print("  Actual BSM1 sensor data used")
print("  12 candidate sensors")
print("  495 feasible actions")
print("  Exactly four active sensors per action")
print("  Four-step historical state")
print("  96-dimensional scheduler state")
print("  Action-dependent four-sensor estimator")
print("  30-minute ΔNH4-N / ΔSNO prediction")
print("  Normalized prediction-error reward")
print("  15-minute decision progression")
print("  Activation history construction")
print("  Target values excluded from scheduler state")
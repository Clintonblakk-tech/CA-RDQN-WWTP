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
# REWARD NORMALIZATION
# ============================================================

# Training-period target-change scales.
#
# These are calculated from the training data only.
# They prevent NH4-N and SNO from being combined on
# incompatible numerical scales.

NH4_SCALE = None
SNO_SCALE = None


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


def build_sensor_matrix(model):
    """
    Construct the 12 candidate-sensor matrix.

    Columns correspond exactly to SENSOR_NAMES.
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


def build_history(X, history):
    """
    Construct a flattened history matrix.

    For time t, the row contains:

        X[t-history]
        X[t-history+1]
        ...
        X[t-1]

    Therefore the current observation X[t] is NOT included.
    """

    rows = []

    for t in range(history, len(X)):
        rows.append(
            X[t-history:t].reshape(-1)
        )

    return np.asarray(rows)


def make_mask(combination, n_sensors=12):
    """
    Create a 12-element binary activation mask.

    1 = sensor active
    0 = sensor inactive
    """

    mask = np.zeros(
        n_sensors,
        dtype=np.float32
    )

    mask[list(combination)] = 1.0

    return mask


def extract_selected_history(
    history_row,
    combination,
    history=HISTORY,
    n_sensors=12
):
    """
    Extract the historical measurements belonging only
    to the four sensors selected by the action.

    The history row is organized as:

        time t-4: sensors 1...12
        time t-3: sensors 1...12
        time t-2: sensors 1...12
        time t-1: sensors 1...12
    """

    selected = []

    for h in range(history):

        start = h * n_sensors

        values = history_row[
            start + np.asarray(combination)
        ]

        selected.extend(values)

    return np.asarray(
        selected,
        dtype=np.float64
    )


# ============================================================
# ESTIMATOR
# ============================================================

class FourSensorEstimator:

    def __init__(self):
        self.coefficients = None

    def fit(
        self,
        X_history,
        delta_targets
    ):
        """
        Fit the estimator using selected historical
        sensor measurements.

        X_history:
            shape = observations × 16

        delta_targets:
            shape = observations × 2
        """

        A = np.column_stack([
            np.ones(len(X_history)),
            X_history
        ])

        self.coefficients = np.linalg.lstsq(
            A,
            delta_targets,
            rcond=None
        )[0]

    def predict(self, X_history):

        A = np.column_stack([
            np.ones(len(X_history)),
            X_history
        ])

        return A @ self.coefficients


# ============================================================
# ENVIRONMENT
# ============================================================

class SensorSchedulingEnvironment:

    def __init__(
        self,
        sensor_history,
        delta_targets,
        combinations,
        train_end
    ):

        self.sensor_history = sensor_history
        self.delta_targets = delta_targets
        self.combinations = combinations

        self.train_end = train_end

        self.n_actions = len(combinations)
        self.n_sensors = 12
        self.n_targets = 2

        self.current_index = train_end

        self.estimators = {}

        # ----------------------------------------------------
        # Calculate target scales from training data only.
        # ----------------------------------------------------

        training_targets = delta_targets[
            :train_end - HISTORY
        ]

        self.target_scale = np.std(
            training_targets,
            axis=0
        )

        # Prevent division by zero.
        self.target_scale = np.maximum(
            self.target_scale,
            1e-8
        )

    def reset(self):

        self.current_index = self.train_end

        return self._get_state()

    def _get_state(self):

        history_row = self.sensor_history[
            self.current_index - HISTORY
        ]

        # Current scheduler state contains:
        # 1. four historical sensor observations
        # 2. four historical activation masks
        #
        # The initial environment test uses the previous
        # action mask repeated through the history.
        #
        # This will be replaced by the actual action history
        # after transition validation.

        measurement_state = history_row.copy()

        mask_state = np.zeros(
            HISTORY * self.n_sensors,
            dtype=np.float32
        )

        return np.concatenate([
            measurement_state,
            mask_state
        ])

    def action_to_sensors(self, action):

        if action < 0 or action >= self.n_actions:
            raise ValueError(
                f"Invalid action: {action}"
            )

        return self.combinations[action]

    def action_to_mask(self, action):

        combination = self.action_to_sensors(
            action
        )

        return make_mask(
            combination
        )

    def evaluate_action(self, action):

        combination = self.action_to_sensors(
            action
        )

        history_row = self.sensor_history[
            self.current_index - HISTORY
        ]

        selected_history = extract_selected_history(
            history_row,
            combination
        )

        estimator = self.estimators[action]

        prediction = estimator.predict(
            selected_history.reshape(1, -1)
        )[0]

        true_delta = self.delta_targets[
            self.current_index
        ]

        error = prediction - true_delta

        normalized_error = (
            error / self.target_scale
        )

        loss = np.mean(
            normalized_error ** 2
        )

        reward = -loss

        return {
            "action": action,
            "combination": combination,
            "prediction": prediction,
            "target": true_delta,
            "error": error,
            "normalized_error": normalized_error,
            "loss": loss,
            "reward": reward,
        }


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
# BUILD DATA
# ============================================================

X = build_sensor_matrix(model)

Y = np.column_stack([
    model.ys_eff_all[:, 9],  # NH4-N
    model.ys_eff_all[:, 8],  # SNO
])


# ============================================================
# 30-MINUTE CHANGE TARGET
# ============================================================

end = len(X) - LEAD

X = X[:end]

delta_Y = (
    Y[LEAD:]
    - Y[:end]
)


# ============================================================
# HISTORICAL SENSOR STATE
# ============================================================

X_history = build_history(
    X,
    HISTORY
)

D = delta_Y[HISTORY:]


# ============================================================
# ACTION SPACE
# ============================================================

combinations = list(
    itertools.combinations(
        range(12),
        4
    )
)


# ============================================================
# TRAIN ONE ESTIMATOR FOR EACH ACTION
# ============================================================

X_train = X_history[
    :TRAIN_END - HISTORY
]

D_train = D[
    :TRAIN_END - HISTORY
]


print()
print("Training four-sensor estimators...")
print(
    "Number of feasible actions:",
    len(combinations)
)


estimators = {}

for action, combination in enumerate(combinations):

    selected_history = []

    for row in X_train:

        selected_history.append(
            extract_selected_history(
                row,
                combination
            )
        )

    selected_history = np.asarray(
        selected_history
    )

    estimator = FourSensorEstimator()

    estimator.fit(
        selected_history,
        D_train
    )

    estimators[action] = estimator


print("Estimator training complete.")


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

environment = SensorSchedulingEnvironment(
    sensor_history=X_history,
    delta_targets=D,
    combinations=combinations,
    train_end=TRAIN_END
)

environment.estimators = estimators


# ============================================================
# ENVIRONMENT STRUCTURE TEST
# ============================================================

print()
print("=" * 70)
print("DQN ENVIRONMENT STRUCTURE TEST")
print("=" * 70)

print()
print("Number of sensors:", environment.n_sensors)
print("Number of actions:", environment.n_actions)
print("State dimension:", 96)
print("Targets:", "NH4-N, SNO")
print("Forecast horizon:", "30 minutes")
print("Decision interval:", "15 minutes")


# ------------------------------------------------------------
# Test several actions.
# ------------------------------------------------------------

test_actions = [
    0,
    1,
    100,
    250,
    494,
]


print()
print("Testing action validity...")


for action in test_actions:

    combination = environment.action_to_sensors(
        action
    )

    mask = environment.action_to_mask(
        action
    )

    active_count = int(
        np.sum(mask)
    )

    names = [
        SENSOR_NAMES[i]
        for i in combination
    ]

    print()
    print(f"Action {action}:")
    print("  Sensors:", ", ".join(names))
    print("  Active sensors:", active_count)

    if active_count != 4:
        raise RuntimeError(
            "Constraint violation: "
            "action does not activate exactly four sensors."
        )


print()
print("All tested actions satisfy the four-sensor constraint.")


# ============================================================
# TEST ENVIRONMENT EVALUATION
# ============================================================

print()
print("Testing reward calculation...")


environment.reset()

test_action = 0

result = environment.evaluate_action(
    test_action
)


print()
print("Test action:", test_action)

print(
    "Selected sensors:",
    ", ".join(
        SENSOR_NAMES[i]
        for i in result["combination"]
    )
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
    "NH4-N error:",
    f"{result['error'][0]:.6f}"
)

print(
    "SNO error:",
    f"{result['error'][1]:.6f}"
)

print(
    "Reward:",
    f"{result['reward']:.6f}"
)


# ============================================================
# FINAL CHECKS
# ============================================================

if environment.n_actions != 495:
    raise RuntimeError(
        "Expected exactly 495 feasible actions."
    )

if environment.n_sensors != 12:
    raise RuntimeError(
        "Expected exactly 12 candidate sensors."
    )

if len(result["combination"]) != 4:
    raise RuntimeError(
        "Selected action does not contain exactly four sensors."
    )

if not np.isfinite(result["reward"]):
    raise RuntimeError(
        "Reward is not finite."
    )


print()
print("=" * 70)
print("DQN ENVIRONMENT STRUCTURE TEST COMPLETE")
print("=" * 70)
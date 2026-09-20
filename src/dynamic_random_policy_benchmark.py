from itertools import combinations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

import bsm2_python
from bsm2_python.bsm1_ol import BSM1OL


# ============================================================
# STEP 20 — DYNAMIC RANDOM-POLICY BENCHMARK
# ============================================================

# -----------------------------
# Configuration
# -----------------------------
DRY_INFLUENT = str(Path(bsm2_python.__file__).resolve().parent / "data" / "dryinfluent.csv")

N_SENSORS = 12
N_ACTIVE = 4
HISTORY = 4
DECISION_INTERVAL = 1          # BSM1 data are already 15-minutely
FORECAST_HORIZON = 2           # 30 minutes = 2 × 15 min

TRAIN_END = 863
EVAL_START = 863

RANDOM_SEED = 20260916


# -----------------------------
# Candidate sensor definitions
# -----------------------------
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


# -----------------------------
# Load BSM1
# -----------------------------
print("\nLoading BSM1 influent...")

data = np.loadtxt(
    DRY_INFLUENT,
    delimiter=",",
    skiprows=1,
)

model = BSM1OL(
    data_in=data,
    timestep=None,
    evaltime=np.array([8.98958333, 13.98958333]),
)

print("Running BSM1 simulation...")

model.simulate(plot=False)

print("BSM1 simulation completed.")


# -----------------------------
# Extract process states
# -----------------------------
states = np.column_stack(
    [
        model.y_out1_all[:, 7],   # DO R1
        model.y_out2_all[:, 7],   # DO R2
        model.y_out3_all[:, 7],   # DO R3
        model.y_out4_all[:, 7],   # DO R4
        model.y_out5_all[:, 7],   # DO R5

        model.y_out1_all[:, 9],   # NH4 R1
        model.y_out2_all[:, 9],   # NH4 R2
        model.y_out3_all[:, 9],   # NH4 R3
        model.y_out4_all[:, 9],   # NH4 R4
        model.y_out5_all[:, 9],   # NH4 R5

        model.y_out3_all[:, 8],   # SNO R3
        model.y_out5_all[:, 8],   # SNO R5
    ]
)

# Effluent targets
effluent_nh4 = model.ys_eff_all[:, 9]
effluent_sno = model.ys_eff_all[:, 8]

n_steps = len(effluent_nh4)

print(f"Number of time steps: {n_steps}")
print(f"Sensor matrix shape: {states.shape}")


# ============================================================
# Chronological train/evaluation split
# ============================================================

max_valid_t = n_steps - FORECAST_HORIZON

if TRAIN_END >= max_valid_t:
    raise ValueError(
        "TRAIN_END is too large for the available forecast horizon."
    )

print("\nChronological split:")
print(f"Training end index : {TRAIN_END - 1}")
print(f"Evaluation start   : {EVAL_START}")


# ============================================================
# Training-only normalization
# ============================================================

train_sensor_values = states[:TRAIN_END]

sensor_mean = train_sensor_values.mean(axis=0)
sensor_std = train_sensor_values.std(axis=0)

sensor_std[sensor_std == 0.0] = 1.0

states_norm = (states - sensor_mean) / sensor_std


# ============================================================
# Generate all feasible actions
# ============================================================

actions = list(combinations(range(N_SENSORS), N_ACTIVE))

assert len(actions) == 495

print("\nAction-space audit:")
print(f"Number of feasible actions: {len(actions)}")
print(f"Each action activates exactly {N_ACTIVE} sensors.")


# ============================================================
# Generate a dynamic random policy
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

random_action_indices = rng.integers(
    low=0,
    high=len(actions),
    size=n_steps,
)

random_masks = np.zeros(
    (n_steps, N_SENSORS),
    dtype=np.float32,
)

for t in range(n_steps):
    selected = actions[random_action_indices[t]]
    random_masks[t, list(selected)] = 1.0


# ============================================================
# Verify exact-four constraint
# ============================================================

active_counts = random_masks.sum(axis=1)

if not np.all(active_counts == N_ACTIVE):
    raise RuntimeError(
        "Dynamic random policy violated the exact-four constraint."
    )

print("Exact-four constraint verified for every time step.")


# ============================================================
# Build causal dynamic-policy observations
#
# At time t:
#
#   1. Previous four time steps provide historical information.
#   2. The random policy selects exactly four current sensors.
#   3. Only the four currently selected measurements are acquired.
#   4. The estimator receives:
#
#        previous 4 × (12 measurements + 12 masks)
#        +
#        current 12 masked measurements
#        +
#        current 12-sensor action mask
#
#      Total = 96 + 24 = 120 features.
#
# Target:
#   ΔY(t+2) = Y(t+2) - Y(t)
# ============================================================

def build_dynamic_example(t):
    """
    Construct one causal estimator example at decision time t.

    Returns
    -------
    features : ndarray
        120-dimensional estimator input.
    delta_nh4 : float
        30-minute change in effluent NH4-N.
    delta_sno : float
        30-minute change in effluent SNO.
    """

    if t < HISTORY:
        raise ValueError("Insufficient history.")

    if t + FORECAST_HORIZON >= n_steps:
        raise ValueError("Insufficient future horizon.")

    # --------------------------------------------------------
    # Historical acquired measurements and masks
    # --------------------------------------------------------
    historical_measurements = []

    for h in range(t - HISTORY, t):
        mask = random_masks[h]

        measured = states_norm[h] * mask

        historical_measurements.extend(measured)
        historical_measurements.extend(mask)

    historical_measurements = np.asarray(
        historical_measurements,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Current action
    # --------------------------------------------------------
    current_mask = random_masks[t]

    current_measurements = (
        states_norm[t] * current_mask
    )

    current_measurements = np.asarray(
        current_measurements,
        dtype=np.float32,
    )

    current_mask = np.asarray(
        current_mask,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Final estimator input
    # --------------------------------------------------------
    features = np.concatenate(
        [
            historical_measurements,
            current_measurements,
            current_mask,
        ]
    )

    # 4 × (12 + 12) + 12 + 12 = 120
    assert features.shape == (120,)

    # --------------------------------------------------------
    # 30-minute change targets
    # --------------------------------------------------------
    delta_nh4 = (
        effluent_nh4[t + FORECAST_HORIZON]
        - effluent_nh4[t]
    )

    delta_sno = (
        effluent_sno[t + FORECAST_HORIZON]
        - effluent_sno[t]
    )

    return features, delta_nh4, delta_sno


# ============================================================
# Construct training and evaluation datasets
# ============================================================

print("\nBuilding dynamic-policy dataset...")

X_train = []
y_train_nh4 = []
y_train_sno = []

X_eval = []
y_eval_nh4 = []
y_eval_sno = []

for t in range(HISTORY, max_valid_t):

    X, dy_nh4, dy_sno = build_dynamic_example(t)

    if t < TRAIN_END:
        X_train.append(X)
        y_train_nh4.append(dy_nh4)
        y_train_sno.append(dy_sno)

    elif t >= EVAL_START:
        X_eval.append(X)
        y_eval_nh4.append(dy_nh4)
        y_eval_sno.append(dy_sno)


X_train = np.asarray(X_train, dtype=np.float32)
y_train_nh4 = np.asarray(y_train_nh4, dtype=np.float32)
y_train_sno = np.asarray(y_train_sno, dtype=np.float32)

X_eval = np.asarray(X_eval, dtype=np.float32)
y_eval_nh4 = np.asarray(y_eval_nh4, dtype=np.float32)
y_eval_sno = np.asarray(y_eval_sno, dtype=np.float32)


print("\nDataset shapes:")
print(f"X_train       : {X_train.shape}")
print(f"NH4 train     : {y_train_nh4.shape}")
print(f"SNO train     : {y_train_sno.shape}")
print(f"X_eval        : {X_eval.shape}")
print(f"NH4 eval      : {y_eval_nh4.shape}")
print(f"SNO eval      : {y_eval_sno.shape}")


# ============================================================
# Train one shared estimator for NH4-N
# ============================================================

print("\nTraining shared NH4-N estimator...")

nh4_model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=RANDOM_SEED,
)

nh4_model.fit(
    X_train,
    y_train_nh4,
)


# ============================================================
# Train one shared estimator for SNO
# ============================================================

print("Training shared SNO estimator...")

sno_model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=RANDOM_SEED + 1,
)

sno_model.fit(
    X_train,
    y_train_sno,
)


# ============================================================
# Evaluate dynamic random policy
# ============================================================

pred_nh4 = nh4_model.predict(X_eval)
pred_sno = sno_model.predict(X_eval)

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


# ============================================================
# Zero-change baseline
# ============================================================

baseline_nh4 = np.sqrt(
    mean_squared_error(
        y_eval_nh4,
        np.zeros_like(y_eval_nh4),
    )
)

baseline_sno = np.sqrt(
    mean_squared_error(
        y_eval_sno,
        np.zeros_like(y_eval_sno),
    )
)


improvement_nh4 = (
    100.0
    * (baseline_nh4 - rmse_nh4)
    / baseline_nh4
)

improvement_sno = (
    100.0
    * (baseline_sno - rmse_sno)
    / baseline_sno
)


# ============================================================
# Action diversity
# ============================================================

unique_train_actions = len(
    np.unique(
        random_action_indices[HISTORY:TRAIN_END]
    )
)

unique_eval_actions = len(
    np.unique(
        random_action_indices[EVAL_START:max_valid_t]
    )
)


# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 70)
print("STEP 20 — DYNAMIC RANDOM-POLICY BENCHMARK")
print("=" * 70)

print("\nDynamic random policy:")
print(f"  Active sensors per decision : {N_ACTIVE}")
print(f"  Decision interval           : 15 minutes")
print(f"  Forecast horizon            : 30 minutes")
print(f"  History length              : {HISTORY} steps")
print(f"  Estimator input dimension   : {X_train.shape[1]}")

print("\nAction diversity:")
print(f"  Unique training actions     : {unique_train_actions}")
print(f"  Unique evaluation actions   : {unique_eval_actions}")
print(f"  Total feasible actions      : {len(actions)}")

print("\nΔNH4-N:")
print(f"  Zero-change RMSE            : {baseline_nh4:.6f}")
print(f"  Dynamic-random RMSE         : {rmse_nh4:.6f}")
print(f"  Relative improvement       : {improvement_nh4:.2f}%")

print("\nΔSNO:")
print(f"  Zero-change RMSE            : {baseline_sno:.6f}")
print(f"  Dynamic-random RMSE         : {rmse_sno:.6f}")
print(f"  Relative improvement       : {improvement_sno:.2f}%")

print("\nCausal information audit:")

# Verify one evaluation example
test_t = EVAL_START

test_X, _, _ = build_dynamic_example(test_t)

print(f"  Test time index             : {test_t}")
print(f"  Feature dimension            : {test_X.shape[0]}")

print("\n" + "=" * 70)
print("STEP 20 COMPLETED")
print("=" * 70)
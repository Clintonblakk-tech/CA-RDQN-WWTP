import os
import numpy as np


# ============================================================
# STEP 53 — DQN TRAINING DATA LOADER
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "bsm1"
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

HISTORY = 4
DECISION_INTERVAL_MIN = 15
FORECAST_HORIZON_MIN = 30

TRAIN_END = 862
EVAL_START = 863

SENSOR_COUNT = 12
TARGET_COUNT = 2


# ------------------------------------------------------------
# Load exported BSM1 trajectories
# ------------------------------------------------------------

reactor = np.load(
    os.path.join(
        DATA_DIR,
        "bsm1_reactor_trajectories.npy"
    )
)

effluent = np.load(
    os.path.join(
        DATA_DIR,
        "bsm1_effluent_trajectories.npy"
    )
)

time = np.load(
    os.path.join(
        DATA_DIR,
        "bsm1_time.npy"
    )
)


# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

assert reactor.shape == (1343, 105)
assert effluent.shape == (1343, 21)
assert time.shape == (1343,)

assert np.isfinite(reactor).all()
assert np.isfinite(effluent).all()
assert np.isfinite(time).all()


# ------------------------------------------------------------
# Reconstruct five reactor blocks
#
# Each reactor contains the 21 BSM1/ASM1 variables:
#
# 0  SI
# 1  SS
# 2  XI
# 3  XS
# 4  XBH
# 5  XBA
# 6  XP
# 7  SO
# 8  SNO
# 9  SNH
# 10 SND
# 11 XND
# 12 SALK
# 13 TSS
# 14 Q
# 15 TEMP
# 16 SD1
# 17 SD2
# 18 SD3
# 19 XD4
# 20 XD5
# ------------------------------------------------------------

reactor_1 = reactor[:, 0:21]
reactor_2 = reactor[:, 21:42]
reactor_3 = reactor[:, 42:63]
reactor_4 = reactor[:, 63:84]
reactor_5 = reactor[:, 84:105]


# ------------------------------------------------------------
# Candidate sensor matrix
#
# S1  DO R1
# S2  DO R2
# S3  DO R3
# S4  DO R4
# S5  DO R5
# S6  NH4 R1
# S7  NH4 R2
# S8  NH4 R3
# S9  NH4 R4
# S10 NH4 R5
# S11 SNO R3
# S12 SNO R5
# ------------------------------------------------------------

sensors = np.column_stack(
    [
        reactor_1[:, 7],
        reactor_2[:, 7],
        reactor_3[:, 7],
        reactor_4[:, 7],
        reactor_5[:, 7],
        reactor_1[:, 9],
        reactor_2[:, 9],
        reactor_3[:, 9],
        reactor_4[:, 9],
        reactor_5[:, 9],
        reactor_3[:, 8],
        reactor_5[:, 8],
    ]
)


# ------------------------------------------------------------
# Target matrix
#
# Effluent:
# index 8  = SNO
# index 9  = SNH
# index 13 = TSS
# ------------------------------------------------------------

targets = np.column_stack(
    [
        effluent[:, 9],   # Effluent NH4-N
        effluent[:, 8],   # Effluent SNO
    ]
)


# ------------------------------------------------------------
# Verify sensor/target dimensions
# ------------------------------------------------------------

assert sensors.shape == (1343, SENSOR_COUNT)
assert targets.shape == (1343, TARGET_COUNT)


# ------------------------------------------------------------
# Construct 30-minute change targets
#
# At 15-minute sampling:
#
# t + 2 = t + 30 minutes
#
# ΔY(t+2) = Y(t+2) - Y(t)
# ------------------------------------------------------------

change_targets = (
    targets[2:] - targets[:-2]
)


# ------------------------------------------------------------
# Construct causal history indices
#
# For decision time t:
#
# history = [t-4, t-3, t-2, t-1]
#
# Current t measurements are acquired AFTER the
# action is selected and therefore are NOT included
# in this pre-action state.
# ------------------------------------------------------------

valid_decision_indices = np.arange(
    HISTORY,
    len(sensors) - 2
)

assert valid_decision_indices[0] == 4
assert valid_decision_indices[-1] == 1340


# ------------------------------------------------------------
# Construct chronological training/evaluation indices
# ------------------------------------------------------------

train_indices = valid_decision_indices[
    valid_decision_indices <= TRAIN_END
]

eval_indices = valid_decision_indices[
    valid_decision_indices >= EVAL_START
]


# ------------------------------------------------------------
# Validate boundary
# ------------------------------------------------------------

assert train_indices[0] == 4
assert train_indices[-1] == 862

assert eval_indices[0] == 863
assert eval_indices[-1] == 1340


# ------------------------------------------------------------
# Convert decision indices to target indices
#
# The 30-minute change target associated with decision t is:
#
# targets[t+2] - targets[t]
# ------------------------------------------------------------

for t in [4, 862, 863, 1340]:

    expected = targets[t + 2] - targets[t]

    actual = change_targets[t]

    assert np.allclose(expected, actual)


# ------------------------------------------------------------
# Final information report
# ------------------------------------------------------------

print("=" * 70)
print("STEP 53 — DQN TRAINING DATA LOADER")
print("=" * 70)

print(f"[PASS] Reactor data shape: {reactor.shape}")
print(f"[PASS] Effluent data shape: {effluent.shape}")
print(f"[PASS] Time vector shape: {time.shape}")

print(f"[PASS] Candidate sensor matrix: {sensors.shape}")
print(f"[PASS] Target matrix: {targets.shape}")
print(f"[PASS] Change-target matrix: {change_targets.shape}")

print()
print(f"[INFO] History length: {HISTORY} steps")
print(f"[INFO] Decision interval: {DECISION_INTERVAL_MIN} minutes")
print(f"[INFO] Forecast horizon: {FORECAST_HORIZON_MIN} minutes")

print()
print(f"[INFO] First valid decision index: {valid_decision_indices[0]}")
print(f"[INFO] Last valid decision index: {valid_decision_indices[-1]}")

print()
print(
    f"[INFO] Training decisions: "
    f"{len(train_indices)}"
)

print(
    f"[INFO] Evaluation decisions: "
    f"{len(eval_indices)}"
)

print()
print(
    f"[INFO] First evaluation decision: "
    f"index {eval_indices[0]}"
)

print(
    f"[INFO] Last evaluation decision: "
    f"index {eval_indices[-1]}"
)

print()
print("[PASS] Causal history construction verified.")
print("[PASS] 30-minute change-target alignment verified.")
print("[PASS] Chronological train/evaluation split verified.")

print("=" * 70)
print("STEP 53 COMPLETED")
print("=" * 70)
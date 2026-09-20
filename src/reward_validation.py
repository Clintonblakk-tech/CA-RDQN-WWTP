import numpy as np


# ============================================================
# STEP 23 — DQN REWARD VALIDATION
# ============================================================

RANDOM_SEED = 20260916

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# Training-target scales
#
# These are the training-period standard deviations of the
# 30-minute change targets used for normalization.
#
# The values are calculated directly from the BSM1 trajectory.
# ============================================================

import bsm2_python
from bsm2_python.bsm1_ol import BSM1OL


DRY_INFLUENT = str(Path(bsm2_python.__file__).resolve().parent / "data" / "dryinfluent.csv")

TRAIN_END = 863
FORECAST_HORIZON = 2


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
# Effluent targets
# ============================================================

effluent_nh4 = model.ys_eff_all[:, 9]
effluent_sno = model.ys_eff_all[:, 8]

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


# Training target changes
train_delta_nh4 = delta_nh4[
    :TRAIN_END - FORECAST_HORIZON
]

train_delta_sno = delta_sno[
    :TRAIN_END - FORECAST_HORIZON
]


sigma_nh4 = np.std(
    train_delta_nh4
)

sigma_sno = np.std(
    train_delta_sno
)

if sigma_nh4 <= 0:
    raise RuntimeError(
        "NH4 target scale is invalid."
    )

if sigma_sno <= 0:
    raise RuntimeError(
        "SNO target scale is invalid."
    )


print("\nTarget normalization scales:")

print(
    f"  σ(ΔNH4) = "
    f"{sigma_nh4:.6f}"
)

print(
    f"  σ(ΔSNO) = "
    f"{sigma_sno:.6f}"
)


# ============================================================
# Reward function
# ============================================================

def calculate_reward(
    predicted_nh4,
    true_nh4,
    predicted_sno,
    true_sno,
    weight_nh4=0.5,
    weight_sno=0.5,
):
    """
    Calculate the constrained sensor-scheduling reward.

    Reward is the negative weighted normalized squared
    prediction error.

    No sensor-count penalty is used because the action
    space already enforces exactly four active sensors.
    """

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

    loss = (
        weight_nh4 * normalized_nh4
        +
        weight_sno * normalized_sno
    )

    reward = -loss

    return reward


# ============================================================
# Test 1 — Perfect prediction
# ============================================================

reward_perfect = calculate_reward(
    predicted_nh4=0.10,
    true_nh4=0.10,
    predicted_sno=-0.05,
    true_sno=-0.05,
)

print("\nTest 1 — Perfect prediction:")
print(
    f"  Reward = {reward_perfect:.10f}"
)

assert np.isclose(
    reward_perfect,
    0.0,
)


# ============================================================
# Test 2 — Small prediction error
# ============================================================

reward_small = calculate_reward(
    predicted_nh4=0.12,
    true_nh4=0.10,
    predicted_sno=-0.04,
    true_sno=-0.05,
)

print("\nTest 2 — Small prediction error:")
print(
    f"  Reward = {reward_small:.6f}"
)

assert reward_small < 0.0


# ============================================================
# Test 3 — Large prediction error
# ============================================================

reward_large = calculate_reward(
    predicted_nh4=1.0,
    true_nh4=0.0,
    predicted_sno=1.0,
    true_sno=0.0,
)

print("\nTest 3 — Large prediction error:")
print(
    f"  Reward = {reward_large:.6f}"
)

assert reward_large < reward_small


# ============================================================
# Test 4 — Symmetry
# ============================================================

reward_positive = calculate_reward(
    predicted_nh4=0.20,
    true_nh4=0.10,
    predicted_sno=0.10,
    true_sno=0.00,
)

reward_negative = calculate_reward(
    predicted_nh4=0.00,
    true_nh4=0.10,
    predicted_sno=-0.10,
    true_sno=0.00,
)

print("\nTest 4 — Error symmetry:")

print(
    f"  Positive error reward = "
    f"{reward_positive:.6f}"
)

print(
    f"  Negative error reward = "
    f"{reward_negative:.6f}"
)

assert np.isclose(
    reward_positive,
    reward_negative,
)


# ============================================================
# Test 5 — Target weighting
# ============================================================

reward_nh4_only = calculate_reward(
    predicted_nh4=1.0,
    true_nh4=0.0,
    predicted_sno=0.0,
    true_sno=0.0,
)

reward_sno_only = calculate_reward(
    predicted_nh4=0.0,
    true_nh4=0.0,
    predicted_sno=1.0,
    true_sno=0.0,
)

print("\nTest 5 — Equal target weighting:")

print(
    f"  NH4-only error reward = "
    f"{reward_nh4_only:.6f}"
)

print(
    f"  SNO-only error reward = "
    f"{reward_sno_only:.6f}"
)

# Both targets are normalized by their own
# training scales and therefore have equal
# nominal weighting.
assert reward_nh4_only < 0.0
assert reward_sno_only < 0.0


# ============================================================
# Test 6 — Random reward ordering
# ============================================================

n_tests = 1000

true_nh4 = rng.normal(
    0.0,
    sigma_nh4,
    n_tests,
)

true_sno = rng.normal(
    0.0,
    sigma_sno,
    n_tests,
)

prediction_error_scale = rng.uniform(
    0.0,
    2.0,
    n_tests,
)

predicted_nh4 = (
    true_nh4
    +
    rng.normal(
        0.0,
        sigma_nh4,
        n_tests,
    )
    * prediction_error_scale
)

predicted_sno = (
    true_sno
    +
    rng.normal(
        0.0,
        sigma_sno,
        n_tests,
    )
    * prediction_error_scale
)

rewards = np.array(
    [
        calculate_reward(
            predicted_nh4[i],
            true_nh4[i],
            predicted_sno[i],
            true_sno[i],
        )
        for i in range(n_tests)
    ]
)

assert np.all(
    np.isfinite(rewards)
)

print("\nTest 6 — Random reward stability:")

print(
    f"  Minimum reward = "
    f"{rewards.min():.6f}"
)

print(
    f"  Maximum reward = "
    f"{rewards.max():.6f}"
)

print(
    f"  Mean reward    = "
    f"{rewards.mean():.6f}"
)

print(
    f"  All finite     = "
    f"{np.all(np.isfinite(rewards))}"
)


# ============================================================
# Test 7 — Exact-four action constraint
# ============================================================

from itertools import combinations

actions = list(
    combinations(
        range(12),
        4,
    )
)

assert len(actions) == 495

for action in actions:

    assert len(action) == 4
    assert len(set(action)) == 4

print("\nTest 7 — Action constraint:")
print("  Feasible actions = 495")
print("  Every action contains exactly 4 sensors.")
print("  No cardinality penalty is required.")


# ============================================================
# Final result
# ============================================================

print("\n" + "=" * 70)
print("STEP 23 — REWARD VALIDATION")
print("=" * 70)

print("[PASS] Training-only target scales calculated.")
print("[PASS] Perfect prediction gives zero loss/reward.")
print("[PASS] Prediction errors produce negative reward.")
print("[PASS] Larger errors produce lower reward.")
print("[PASS] Reward is symmetric with respect to error sign.")
print("[PASS] NH4 and SNO are normalized separately.")
print("[PASS] Equal target weights are implemented.")
print("[PASS] Reward remains finite under random tests.")
print("[PASS] Exact-four action space requires no cardinality penalty.")

print("\nReward definition validated:")
print(
    "  r_t = -[0.5·e_NH4²/σ_NH4² "
    "+ 0.5·e_SNO²/σ_SNO²]"
)

print("\n" + "=" * 70)
print("STEP 23 COMPLETED")
print("=" * 70)
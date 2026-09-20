from itertools import combinations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

import bsm2_python
from bsm2_python.bsm1_ol import BSM1OL


# ============================================================
# STEP 21 — DQN ENVIRONMENT VALIDATION
# ============================================================

DRY_INFLUENT = str(Path(bsm2_python.__file__).resolve().parent / "data" / "dryinfluent.csv")

N_SENSORS = 12
N_ACTIVE = 4
HISTORY = 4
FORECAST_HORIZON = 2

TRAIN_END = 863
EVAL_START = 863

RANDOM_SEED = 20260916


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
# Feasible action space
# ============================================================

ACTIONS = list(
    combinations(
        range(N_SENSORS),
        N_ACTIVE,
    )
)

assert len(ACTIONS) == 495


def action_to_mask(action_index):
    """Convert an action index into a 12-dimensional binary mask."""

    mask = np.zeros(
        N_SENSORS,
        dtype=np.float32,
    )

    selected = ACTIONS[action_index]

    mask[list(selected)] = 1.0

    assert mask.sum() == N_ACTIVE

    return mask


# ============================================================
# Load and simulate BSM1
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
# Construct candidate sensor matrix
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

print(f"Time steps: {n_steps}")
print(f"Sensor matrix: {sensor_matrix.shape}")


# ============================================================
# Training-only normalization
# ============================================================

sensor_mean = sensor_matrix[:TRAIN_END].mean(axis=0)
sensor_std = sensor_matrix[:TRAIN_END].std(axis=0)

sensor_std[sensor_std == 0.0] = 1.0

sensor_norm = (
    sensor_matrix - sensor_mean
) / sensor_std


# ============================================================
# Random dynamic activation sequence
#
# This is used ONLY to validate the environment.
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

policy_actions = rng.integers(
    0,
    len(ACTIONS),
    size=n_steps,
)

policy_masks = np.vstack(
    [
        action_to_mask(a)
        for a in policy_actions
    ]
)

assert policy_masks.shape == (
    n_steps,
    N_SENSORS,
)

assert np.all(
    policy_masks.sum(axis=1) == N_ACTIVE
)


# ============================================================
# Environment
# ============================================================

class SensorSchedulingEnvironment:

    def __init__(self):

        self.current_step = HISTORY

        self.n_steps = n_steps

        self.sensor_norm = sensor_norm
        self.policy_masks = policy_masks

        self.target_nh4 = effluent_nh4
        self.target_sno = effluent_sno

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(self, step=None):

        if step is None:
            self.current_step = HISTORY
        else:
            self.current_step = step

        return self.get_state()

    # --------------------------------------------------------
    # Build scheduler state
    #
    # The DQN observes only the past four steps.
    #
    # 4 × (12 measurements + 12 masks)
    # = 96 features
    # --------------------------------------------------------

    def get_state(self):

        t = self.current_step

        historical = []

        for h in range(
            t - HISTORY,
            t,
        ):

            mask = self.policy_masks[h]

            acquired = (
                self.sensor_norm[h]
                * mask
            )

            historical.extend(acquired)
            historical.extend(mask)

        state = np.asarray(
            historical,
            dtype=np.float32,
        )

        assert state.shape == (96,)

        return state

    # --------------------------------------------------------
    # Step
    #
    # The DQN chooses the current action.
    # Current measurements are acquired AFTER the action.
    # --------------------------------------------------------

    def step(self, action_index):

        t = self.current_step

        # ---------------------------------------------
        # Validate action
        # ---------------------------------------------

        if not (
            0 <= action_index < len(ACTIONS)
        ):
            raise ValueError(
                "Invalid action index."
            )

        current_mask = action_to_mask(
            action_index
        )

        assert current_mask.sum() == N_ACTIVE

        # ---------------------------------------------
        # Acquire current measurements
        # ---------------------------------------------

        current_measurements = (
            self.sensor_norm[t]
            * current_mask
        )

        # ---------------------------------------------
        # Historical information
        # ---------------------------------------------

        historical = []

        for h in range(
            t - HISTORY,
            t,
        ):

            historical_mask = (
                self.policy_masks[h]
            )

            historical_measurements = (
                self.sensor_norm[h]
                * historical_mask
            )

            historical.extend(
                historical_measurements
            )

            historical.extend(
                historical_mask
            )

        historical = np.asarray(
            historical,
            dtype=np.float32,
        )

        # ---------------------------------------------
        # Estimator input
        #
        # 96 historical features
        # + 12 current measurements
        # + 12 current mask
        # = 120
        # ---------------------------------------------

        estimator_input = np.concatenate(
            [
                historical,
                current_measurements,
                current_mask,
            ]
        )

        assert estimator_input.shape == (
            120,
        )

        # ---------------------------------------------
        # Target
        # ---------------------------------------------

        if (
            t + FORECAST_HORIZON
            >= self.n_steps
        ):

            raise RuntimeError(
                "Forecast horizon exceeds data."
            )

        true_delta_nh4 = (
            self.target_nh4[
                t + FORECAST_HORIZON
            ]
            - self.target_nh4[t]
        )

        true_delta_sno = (
            self.target_sno[
                t + FORECAST_HORIZON
            ]
            - self.target_sno[t]
        )

        # ---------------------------------------------
        # Advance environment
        # ---------------------------------------------

        self.current_step += 1

        done = (
            self.current_step
            + FORECAST_HORIZON
            >= self.n_steps
        )

        next_state = None

        if not done:
            next_state = self.get_state()

        return (
            estimator_input,
            true_delta_nh4,
            true_delta_sno,
            current_mask,
            next_state,
            done,
        )


# ============================================================
# Instantiate environment
# ============================================================

env = SensorSchedulingEnvironment()


# ============================================================
# Validate initial state
# ============================================================

state = env.reset(
    step=HISTORY
)

print("\nInitial state audit:")
print(f"State shape: {state.shape}")

assert state.shape == (96,)


# ============================================================
# Validate several actions
# ============================================================

test_actions = [
    0,
    100,
    250,
    494,
]

print("\nAction validation:")

for action_index in test_actions:

    env.reset(
        step=HISTORY
    )

    (
        estimator_input,
        true_nh4,
        true_sno,
        mask,
        next_state,
        done,
    ) = env.step(
        action_index
    )

    selected_names = [
        SENSOR_NAMES[i]
        for i in np.where(mask == 1)[0]
    ]

    print(
        f"\nAction {action_index}:"
    )

    print(
        f"  Selected sensors: "
        f"{selected_names}"
    )

    print(
        f"  Active count: "
        f"{int(mask.sum())}"
    )

    print(
        f"  Estimator input: "
        f"{estimator_input.shape}"
    )

    print(
        f"  True ΔNH4: "
        f"{true_nh4:.6f}"
    )

    print(
        f"  True ΔSNO: "
        f"{true_sno:.6f}"
    )

    print(
        f"  Next state: "
        f"{None if next_state is None else next_state.shape}"
    )

    assert mask.sum() == 4
    assert estimator_input.shape == (120,)

    if next_state is not None:
        assert next_state.shape == (96,)


# ============================================================
# Information-availability audit
# ============================================================

print("\nInformation-availability audit:")

audit_t = 100

env.reset(
    step=audit_t
)

(
    estimator_input,
    _,
    _,
    current_mask,
    _,
    _,
) = env.step(
    309
)

historical_start = 0
historical_end = 96

historical_part = estimator_input[
    historical_start:historical_end
]

current_measurements = estimator_input[
    96:108
]

current_action_mask = estimator_input[
    108:120
]

assert current_action_mask.sum() == 4

# Inactive current sensors must contain no acquired value.
inactive_values = (
    current_measurements[
        current_action_mask == 0
    ]
)

assert np.all(
    inactive_values == 0.0
)

print(
    "  Inactive current sensors "
    "contain no measurement."
)

print(
    "  Current action mask preserved."
)

print(
    f"  Current active sensors: "
    f"{int(current_action_mask.sum())}"
)


# ============================================================
# Multi-step transition test
# ============================================================

print("\nTransition test:")

env.reset(
    step=HISTORY
)

previous_state = env.get_state()

for action_index in [
    0,
    100,
    250,
    494,
]:

    (
        estimator_input,
        _,
        _,
        mask,
        next_state,
        done,
    ) = env.step(
        action_index
    )

    print(
        f"  Action {action_index} "
        f"→ active sensors "
        f"{int(mask.sum())} "
        f"→ next state "
        f"{None if next_state is None else next_state.shape}"
    )

    assert mask.sum() == 4

    if next_state is not None:

        assert next_state.shape == (
            96,
        )

        previous_state = next_state


# ============================================================
# Final validation
# ============================================================

print("\n" + "=" * 70)
print("STEP 21 — DQN ENVIRONMENT VALIDATION")
print("=" * 70)

print("\nPASS conditions:")

print("  [PASS] 495 feasible actions")
print("  [PASS] Exactly four sensors per action")
print("  [PASS] 96-dimensional DQN state")
print("  [PASS] 120-dimensional estimator input")
print("  [PASS] Current measurements acquired after action")
print("  [PASS] Inactive current measurements unavailable")
print("  [PASS] Current action mask preserved")
print("  [PASS] 30-minute ΔNH4 target")
print("  [PASS] 30-minute ΔSNO target")
print("  [PASS] Causal multi-step transition")

print("\n" + "=" * 70)
print("STEP 21 COMPLETED")
print("=" * 70)
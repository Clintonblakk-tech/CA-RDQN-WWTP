import itertools
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

N_SENSORS = 12
HISTORY = 4
STATE_DIM = 96


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
# ACTION SPACE
# ============================================================

combinations = list(
    itertools.combinations(
        range(N_SENSORS),
        4
    )
)


def make_mask(combination):
    mask = np.zeros(
        N_SENSORS,
        dtype=np.float32
    )

    mask[list(combination)] = 1.0

    return mask


# ============================================================
# SENSOR MEASUREMENT GENERATION
# ============================================================

def make_measurements(time_index):
    """
    Create deterministic test measurements.

    The values are deliberately distinct at each time step
    so that temporal ordering can be verified.
    """

    return (
        np.arange(
            N_SENSORS,
            dtype=np.float32
        )
        + 100.0 * time_index
    )


# ============================================================
# RL ENVIRONMENT
# ============================================================

class RLTransitionEnvironment:

    def __init__(
        self,
        history=HISTORY,
        n_sensors=N_SENSORS
    ):

        self.history = history
        self.n_sensors = n_sensors

        self.measurement_history = []
        self.mask_history = []

        self.time_index = 0

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        self.time_index = 0

        initial_measurements = make_measurements(
            self.time_index
        )

        zero_mask = np.zeros(
            self.n_sensors,
            dtype=np.float32
        )

        self.measurement_history = [
            initial_measurements.copy()
            for _ in range(self.history)
        ]

        self.mask_history = [
            zero_mask.copy()
            for _ in range(self.history)
        ]

        return self._get_state()

    # --------------------------------------------------------
    # STATE
    # --------------------------------------------------------

    def _get_state(self):

        measurement_state = np.concatenate(
            self.measurement_history
        )

        mask_state = np.concatenate(
            self.mask_history
        )

        state = np.concatenate([
            measurement_state,
            mask_state
        ])

        return state

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    def step(self, action):

        if action < 0 or action >= len(combinations):

            raise ValueError(
                f"Invalid action: {action}"
            )

        combination = combinations[action]

        mask = make_mask(
            combination
        )

        # ----------------------------------------------------
        # Reward for this isolated transition test.
        #
        # The reward is deliberately simple because this
        # test is checking transition mechanics, not estimator
        # performance.
        # ----------------------------------------------------

        reward = -float(
            np.sum(
                (mask - 0.5) ** 2
            )
        )

        # ----------------------------------------------------
        # Advance one decision interval.
        #
        # One RL step = 15 minutes.
        # ----------------------------------------------------

        self.time_index += 1

        new_measurements = make_measurements(
            self.time_index
        )

        # ----------------------------------------------------
        # Append the newly observed measurements and the
        # action selected at the preceding decision.
        # ----------------------------------------------------

        self.measurement_history.append(
            new_measurements.copy()
        )

        self.mask_history.append(
            mask.copy()
        )

        # Retain only the most recent four observations.
        self.measurement_history = (
            self.measurement_history[-self.history:]
        )

        self.mask_history = (
            self.mask_history[-self.history:]
        )

        next_state = self._get_state()

        done = False

        info = {
            "action": action,
            "combination": combination,
            "mask": mask,
            "time_index": self.time_index,
        }

        return (
            next_state,
            reward,
            done,
            info,
        )


# ============================================================
# INITIALIZE ENVIRONMENT
# ============================================================

environment = RLTransitionEnvironment()

state_0 = environment.reset()


print("=" * 70)
print("RL TRANSITION TEST")
print("=" * 70)

print()
print("Number of sensors:", N_SENSORS)
print("Number of actions:", len(combinations))
print("History length:", HISTORY)
print("State dimension:", len(state_0))
print("Decision interval:", "15 minutes")


# ============================================================
# INITIAL STATE CHECK
# ============================================================

if len(state_0) != STATE_DIM:

    raise RuntimeError(
        "Initial state dimension is not 96."
    )


print()
print("Initial state dimension verified.")


# ============================================================
# ACTION SEQUENCE
# ============================================================

actions = [
    0,
    100,
    250,
    494,
]


previous_state = state_0


for step_number, action in enumerate(
    actions,
    start=1
):

    next_state, reward, done, info = (
        environment.step(action)
    )

    combination = info["combination"]
    mask = info["mask"]

    print()
    print(
        f"RL STEP {step_number}"
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
        "Reward:",
        f"{reward:.6f}"
    )

    print(
        "Next time index:",
        info["time_index"]
    )

    print(
        "Next state dimension:",
        len(next_state)
    )

    # --------------------------------------------------------
    # Check action constraint.
    # --------------------------------------------------------

    if int(mask.sum()) != 4:

        raise RuntimeError(
            "Exactly-four-sensor constraint violated."
        )

    # --------------------------------------------------------
    # Check state dimension.
    # --------------------------------------------------------

    if len(next_state) != STATE_DIM:

        raise RuntimeError(
            "Next state dimension is not 96."
        )

    previous_state = next_state


# ============================================================
# VERIFY FINAL MEASUREMENT HISTORY
# ============================================================

expected_measurements = np.concatenate([
    make_measurements(1),
    make_measurements(2),
    make_measurements(3),
    make_measurements(4),
])

actual_measurements = previous_state[:48]


print()
print("Verifying final measurement history...")


if not np.array_equal(
    actual_measurements,
    expected_measurements
):

    raise RuntimeError(
        "Final measurement history is incorrect."
    )


print(
    "Final measurement history verified."
)


# ============================================================
# VERIFY FINAL ACTION HISTORY
# ============================================================

expected_masks = np.concatenate([
    make_mask(combinations[0]),
    make_mask(combinations[100]),
    make_mask(combinations[250]),
    make_mask(combinations[494]),
])

actual_masks = previous_state[48:]


print()
print("Verifying final activation history...")


if not np.array_equal(
    actual_masks,
    expected_masks
):

    raise RuntimeError(
        "Final activation-mask history is incorrect."
    )


print(
    "Final activation-mask history verified."
)


# ============================================================
# VERIFY TEMPORAL TRANSITION
# ============================================================

print()
print("Verifying temporal progression...")

expected_final_time = len(actions)

if environment.time_index != expected_final_time:

    raise RuntimeError(
        "Environment time index did not advance correctly."
    )


print(
    "Time progression verified:"
)

print(
    f"  Initial decision index: 0"
)

print(
    f"  Final decision index: {environment.time_index}"
)

print(
    f"  Decision intervals elapsed: {environment.time_index}"
)

print(
    f"  Equivalent elapsed time: "
    f"{environment.time_index * 15} minutes"
)


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 70)
print("RL TRANSITION TEST PASSED")
print("=" * 70)

print()
print("Verified:")
print("  State dimension = 96")
print("  Action space = 495")
print("  Exactly 4 sensors per action")
print("  Measurement history advances correctly")
print("  Activation history advances correctly")
print("  One RL step = one 15-minute decision interval")
print("  State transition ordering is correct")
import itertools
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

N_SENSORS = 12
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
# ACTION SPACE
# ============================================================

combinations = list(
    itertools.combinations(
        range(N_SENSORS),
        4
    )
)


def make_mask(combination):
    """
    Convert a four-sensor combination into a
    12-element binary activation mask.
    """

    mask = np.zeros(
        N_SENSORS,
        dtype=np.float32
    )

    mask[list(combination)] = 1.0

    return mask


# ============================================================
# STATE HISTORY
# ============================================================

class StateHistory:

    def __init__(
        self,
        history=HISTORY,
        n_sensors=N_SENSORS
    ):

        self.history = history
        self.n_sensors = n_sensors

        self.measurement_history = []

        self.mask_history = []

    def reset(
        self,
        initial_measurements
    ):

        self.measurement_history = [
            np.asarray(initial_measurements, dtype=np.float32)
            for _ in range(self.history)
        ]

        # Before the first action, no sensors have
        # previously been activated.

        zero_mask = np.zeros(
            self.n_sensors,
            dtype=np.float32
        )

        self.mask_history = [
            zero_mask.copy()
            for _ in range(self.history)
        ]

    def append(
        self,
        measurements,
        mask
    ):

        self.measurement_history.append(
            np.asarray(
                measurements,
                dtype=np.float32
            )
        )

        self.mask_history.append(
            np.asarray(
                mask,
                dtype=np.float32
            )
        )

        self.measurement_history = (
            self.measurement_history[-self.history:]
        )

        self.mask_history = (
            self.mask_history[-self.history:]
        )

    def get_state(self):

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


# ============================================================
# CREATE SYNTHETIC TEST MEASUREMENTS
# ============================================================

# We use clearly distinguishable values so that it is
# possible to verify the state ordering manually.

measurements_t0 = np.arange(
    12,
    dtype=np.float32
)

measurements_t1 = (
    np.arange(
        12,
        dtype=np.float32
    )
    + 100
)

measurements_t2 = (
    np.arange(
        12,
        dtype=np.float32
    )
    + 200
)

measurements_t3 = (
    np.arange(
        12,
        dtype=np.float32
    )
    + 300
)

measurements_t4 = (
    np.arange(
        12,
        dtype=np.float32
    )
    + 400
)


# ============================================================
# DEFINE TEST ACTIONS
# ============================================================

action_A = 0

action_B = 100

action_C = 250

action_D = 494


combination_A = combinations[action_A]
combination_B = combinations[action_B]
combination_C = combinations[action_C]
combination_D = combinations[action_D]


mask_A = make_mask(combination_A)
mask_B = make_mask(combination_B)
mask_C = make_mask(combination_C)
mask_D = make_mask(combination_D)


# ============================================================
# INITIALIZE STATE HISTORY
# ============================================================

history = StateHistory()

history.reset(
    measurements_t0
)


# ============================================================
# INITIAL STATE TEST
# ============================================================

state = history.get_state()


print("=" * 70)
print("STATE TRANSITION TEST")
print("=" * 70)

print()
print("Number of sensors:", N_SENSORS)
print("History length:", HISTORY)
print("Expected measurement features:", 48)
print("Expected mask features:", 48)
print("Expected total state dimension:", 96)

print()
print("Initial state dimension:", len(state))


if len(state) != 96:
    raise RuntimeError(
        "Initial state dimension is not 96."
    )


# ============================================================
# ACTION A
# ============================================================

history.append(
    measurements_t1,
    mask_A
)

state_A = history.get_state()


print()
print("ACTION A")
print("Sensors:")

for index in combination_A:
    print(" ", SENSOR_NAMES[index])

print(
    "Active sensors:",
    int(mask_A.sum())
)

print(
    "State dimension:",
    len(state_A)
)


# ============================================================
# ACTION B
# ============================================================

history.append(
    measurements_t2,
    mask_B
)

state_B = history.get_state()


print()
print("ACTION B")
print("Sensors:")

for index in combination_B:
    print(" ", SENSOR_NAMES[index])

print(
    "Active sensors:",
    int(mask_B.sum())
)

print(
    "State dimension:",
    len(state_B)
)


# ============================================================
# ACTION C
# ============================================================

history.append(
    measurements_t3,
    mask_C
)

state_C = history.get_state()


print()
print("ACTION C")
print("Sensors:")

for index in combination_C:
    print(" ", SENSOR_NAMES[index])

print(
    "Active sensors:",
    int(mask_C.sum())
)

print(
    "State dimension:",
    len(state_C)
)


# ============================================================
# ACTION D
# ============================================================

history.append(
    measurements_t4,
    mask_D
)

state_D = history.get_state()


print()
print("ACTION D")
print("Sensors:")

for index in combination_D:
    print(" ", SENSOR_NAMES[index])

print(
    "Active sensors:",
    int(mask_D.sum())
)

print(
    "State dimension:",
    len(state_D)
)


# ============================================================
# VERIFY MASK HISTORY
# ============================================================

expected_masks = np.concatenate([
    mask_A,
    mask_B,
    mask_C,
    mask_D
])

actual_masks = state_D[48:]


print()
print("Verifying final four-step activation history...")


if not np.array_equal(
    actual_masks,
    expected_masks
):

    raise RuntimeError(
        "Activation-mask history is incorrect."
    )


print(
    "Final mask history:",
    actual_masks.astype(int)
)


# ============================================================
# VERIFY EACH MASK CONTAINS EXACTLY FOUR ACTIVE SENSORS
# ============================================================

for name, mask in [
    ("A", mask_A),
    ("B", mask_B),
    ("C", mask_C),
    ("D", mask_D),
]:

    active = int(
        mask.sum()
    )

    if active != 4:

        raise RuntimeError(
            f"Action {name} does not contain exactly "
            f"four active sensors."
        )


# ============================================================
# VERIFY MEASUREMENT HISTORY
# ============================================================

expected_measurements = np.concatenate([
    measurements_t1,
    measurements_t2,
    measurements_t3,
    measurements_t4
])

actual_measurements = state_D[:48]


print()
print("Verifying final four-step measurement history...")


if not np.array_equal(
    actual_measurements,
    expected_measurements
):

    raise RuntimeError(
        "Measurement history is incorrect."
    )


print(
    "Final measurement history verified."
)


# ============================================================
# FINAL STATE CHECK
# ============================================================

if len(state_D) != 96:

    raise RuntimeError(
        "Final state dimension is not 96."
    )


print()
print("=" * 70)
print("STATE TRANSITION TEST PASSED")
print("=" * 70)

print()
print(
    "The state correctly contains:"
)

print(
    "  48 historical measurement features"
)

print(
    "  48 historical activation-mask features"
)

print(
    "  Total = 96 state features"
)

print()
print(
    "Four-sensor constraint verified for every test action."
)

print(
    "Activation history ordering verified."
)

print(
    "Measurement history ordering verified."
)

print()
print("=" * 70)
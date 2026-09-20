import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

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
# MASKING FUNCTION
# ============================================================

def apply_sensor_mask(
    measurements,
    mask
):
    """
    Return the measurements available to the estimator.

    Active sensor:
        retain its measurement.

    Inactive sensor:
        measurement becomes zero.

    The separate mask tells the estimator whether zero
    represents an unavailable measurement or a genuine zero.
    """

    measurements = np.asarray(
        measurements,
        dtype=np.float32
    )

    mask = np.asarray(
        mask,
        dtype=np.float32
    )

    if len(measurements) != N_SENSORS:
        raise ValueError(
            "Expected 12 measurements."
        )

    if len(mask) != N_SENSORS:
        raise ValueError(
            "Expected 12 mask values."
        )

    masked_measurements = (
        measurements * mask
    )

    return masked_measurements


# ============================================================
# TEST MEASUREMENTS
# ============================================================

# Deliberately distinct values.
measurements = np.array(
    [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
    ],
    dtype=np.float32
)


# ============================================================
# TEST ACTION
# ============================================================

# Activate:
#
# S1  = DO_R1
# S5  = DO_R5
# S8  = NH4_R3
# S12 = SNO_R5

active_indices = [
    0,
    4,
    7,
    11,
]


mask = np.zeros(
    N_SENSORS,
    dtype=np.float32
)

mask[active_indices] = 1.0


# ============================================================
# APPLY MASK
# ============================================================

masked = apply_sensor_mask(
    measurements,
    mask
)


print("=" * 70)
print("MASKED SENSOR AVAILABILITY TEST")
print("=" * 70)

print()
print("Original measurements:")
print(measurements)

print()
print("Activation mask:")
print(mask.astype(int))

print()
print("Masked measurements:")
print(masked)


# ============================================================
# VERIFY ACTIVE SENSORS
# ============================================================

print()
print("Checking active sensors...")


for index in active_indices:

    if masked[index] != measurements[index]:

        raise RuntimeError(
            f"Active sensor {SENSOR_NAMES[index]} "
            "was incorrectly masked."
        )


print(
    "Active sensor measurements preserved: VERIFIED"
)


# ============================================================
# VERIFY INACTIVE SENSORS
# ============================================================

print()
print("Checking inactive sensors...")


for index in range(N_SENSORS):

    if index not in active_indices:

        if masked[index] != 0.0:

            raise RuntimeError(
                f"Inactive sensor {SENSOR_NAMES[index]} "
                "was not masked."
            )


print(
    "Inactive sensor measurements removed: VERIFIED"
)


# ============================================================
# VERIFY EXACTLY FOUR ACTIVE SENSORS
# ============================================================

active_count = int(
    mask.sum()
)


print()
print(
    "Active sensor count:",
    active_count
)


if active_count != 4:

    raise RuntimeError(
        "The test action does not contain exactly "
        "four active sensors."
    )


# ============================================================
# VERIFY MASK DISAMBIGUATES ZERO
# ============================================================

print()
print("Testing zero-value ambiguity...")


# Create a genuine zero measurement for an active sensor.
measurements_with_zero = measurements.copy()

measurements_with_zero[0] = 0.0

masked_zero = apply_sensor_mask(
    measurements_with_zero,
    mask
)


# S1 is active and genuinely measured zero.
if masked_zero[0] != 0.0:

    raise RuntimeError(
        "Genuine zero measurement was altered."
    )


# But S2 is inactive.
# Its zero also results from masking.
# The mask distinguishes the two situations.

if mask[0] != 1.0:
    raise RuntimeError(
        "Active zero-valued sensor has incorrect mask."
    )

if mask[1] != 0.0:
    raise RuntimeError(
        "Inactive sensor has incorrect mask."
    )


print(
    "Measurement/mask ambiguity handling: VERIFIED"
)


# ============================================================
# CONSTRUCT ONE STATE
# ============================================================

# The scheduler/estimator interface uses:
#
# 12 masked measurements
# +
# 12 availability indicators
#
# for each timestep.

state_t = np.concatenate([
    masked,
    mask,
])


print()
print(
    "Single-timestep state dimension:",
    len(state_t)
)


if len(state_t) != 24:

    raise RuntimeError(
        "Single-timestep state must contain "
        "24 features."
    )


# ============================================================
# FOUR-STEP STATE
# ============================================================

history = np.tile(
    state_t,
    4
)


print(
    "Four-step state dimension:",
    len(history)
)


if len(history) != 96:

    raise RuntimeError(
        "Four-step state must contain "
        "96 features."
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("MASKED SENSOR AVAILABILITY TEST PASSED")
print("=" * 70)

print()
print("Verified:")
print("  Active measurements are retained.")
print("  Inactive measurements are masked.")
print("  The activation mask identifies availability.")
print("  Exactly four sensors are active.")
print("  Genuine zero measurements remain valid.")
print("  One timestep = 24 features.")
print("  Four timesteps = 96 features.")
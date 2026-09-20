import itertools
import numpy as np


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

N_SENSORS = len(SENSOR_NAMES)
HISTORY = 4


# ============================================================
# ACTION SPACE
# ============================================================

ACTIONS = list(
    itertools.combinations(range(N_SENSORS), 4)
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("CORRECTED ESTIMATOR INFORMATION INTERFACE TEST")
print("=" * 70)

print(f"\nCandidate sensors: {N_SENSORS}")
print(f"Feasible actions: {len(ACTIONS)}")
print(f"History length: {HISTORY}")


# ============================================================
# SYNTHETIC SENSOR HISTORY
#
# Each sensor has a unique value at each historical time.
# ============================================================

sensor_history = np.zeros(
    (HISTORY, N_SENSORS),
    dtype=float
)

for t in range(HISTORY):

    for sensor_index in range(N_SENSORS):

        sensor_history[t, sensor_index] = (
            1000 * t + sensor_index
        )


# ============================================================
# HISTORICAL ACTIONS
#
# These represent the sensors that were ACTUALLY active at
# each historical decision time.
# ============================================================

historical_actions = [
    0,
    100,
    250,
    494,
]


# ============================================================
# HISTORICAL ACTIVATION MASKS
# ============================================================

activation_history = np.zeros(
    (HISTORY, N_SENSORS),
    dtype=int
)

for t, action_id in enumerate(historical_actions):

    selected_sensors = ACTIONS[action_id]

    activation_history[
        t,
        list(selected_sensors)
    ] = 1


# ============================================================
# ACQUIRED MEASUREMENTS
#
# A measurement exists for the estimator ONLY when the
# corresponding sensor was active.
#
# NaN represents "not acquired" in this test.
# ============================================================

acquired_history = sensor_history.copy()

acquired_history[
    activation_history == 0
] = np.nan


print("\nHistorical acquired measurements:")
print("--------------------------------")

print(acquired_history)


# ============================================================
# CURRENT ACTION
# ============================================================

current_action = 250

current_selected = ACTIONS[current_action]

print("\nCurrent action:")
print(f"Action ID: {current_action}")

print(
    "Selected sensors:",
    [
        SENSOR_NAMES[index]
        for index in current_selected
    ]
)


# ============================================================
# CORRECTED ESTIMATOR INPUT
#
# The estimator receives TWO pieces of information:
#
# 1. Acquired measurements
# 2. Availability masks
#
# It does NOT receive the underlying values of sensors that
# were inactive.
# ============================================================

measurement_input = np.zeros(
    (HISTORY, N_SENSORS),
    dtype=float
)

availability_input = activation_history.copy()


for t in range(HISTORY):

    for sensor_index in range(N_SENSORS):

        if availability_input[t, sensor_index] == 1:

            measurement_input[
                t,
                sensor_index
            ] = sensor_history[t, sensor_index]

        else:

            measurement_input[
                t,
                sensor_index
            ] = 0.0


# ============================================================
# COMBINED ESTIMATOR INPUT
#
# First 12 columns:
#     masked sensor measurements
#
# Next 12 columns:
#     availability masks
#
# Therefore:
#
# 4 historical steps × 24 features = 96 features
# ============================================================

corrected_state = np.concatenate(
    [
        measurement_input,
        availability_input.astype(float)
    ],
    axis=1
)


corrected_state = corrected_state.flatten()


print("\nCorrected estimator state:")
print("--------------------------------")

print(
    f"State shape: {corrected_state.shape}"
)

print(
    "Expected state dimension: 96"
)


# ============================================================
# INFORMATION-AVAILABILITY VERIFICATION
# ============================================================

violations = []

for t in range(HISTORY):

    for sensor_index in range(N_SENSORS):

        measurement = measurement_input[
            t,
            sensor_index
        ]

        available = (
            availability_input[t, sensor_index] == 1
        )

        true_value = sensor_history[
            t,
            sensor_index
        ]

        # If the sensor was unavailable, the actual sensor
        # value must NOT appear in the estimator measurement
        # input.

        if not available:

            if measurement == true_value:

                violations.append(
                    (
                        t,
                        sensor_index,
                        SENSOR_NAMES[sensor_index],
                        true_value
                    )
                )


# ============================================================
# AVAILABILITY-MASK VERIFICATION
# ============================================================

mask_errors = []

for t in range(HISTORY):

    for sensor_index in range(N_SENSORS):

        expected = activation_history[
            t,
            sensor_index
        ]

        actual = availability_input[
            t,
            sensor_index
        ]

        if expected != actual:

            mask_errors.append(
                (
                    t,
                    sensor_index,
                    expected,
                    actual
                )
            )


# ============================================================
# CURRENT ACTION AVAILABILITY TEST
#
# For the current four selected sensors, determine exactly
# which historical measurements were actually available.
# ============================================================

print("\nCurrent-action historical availability:")
print("----------------------------------------")

for sensor_index in current_selected:

    history_status = []

    for t in range(HISTORY):

        if activation_history[
            t,
            sensor_index
        ] == 1:

            history_status.append("AVAILABLE")

        else:

            history_status.append("UNAVAILABLE")

    print(
        f"{SENSOR_NAMES[sensor_index]}: "
        f"{history_status}"
    )


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("INFORMATION-AVAILABILITY TEST")
print("=" * 70)


if violations:

    print(
        "FAILED: unavailable sensor values entered "
        "the estimator measurement input."
    )

elif mask_errors:

    print(
        "FAILED: availability masks do not match "
        "the actual activation history."
    )

elif corrected_state.shape != (HISTORY * N_SENSORS * 2,):

    print(
        "FAILED: corrected state has an unexpected dimension."
    )

else:

    print(
        "PASSED: unavailable sensor values are excluded."
    )

    print(
        "PASSED: historical availability masks are preserved."
    )

    print(
        "PASSED: measurement and availability information "
        "are represented separately."
    )

    print(
        "PASSED: corrected state dimension = 96."
    )


# ============================================================
# FINAL ARCHITECTURAL STATEMENT
# ============================================================

print("\n" + "=" * 70)

if not violations and not mask_errors:

    print(
        "CORRECTED ESTIMATOR INTERFACE: VERIFIED"
    )

    print("=" * 70)

    print(
        "\nThe estimator can now distinguish between:"
    )

    print(
        "  1. A sensor measurement that was actually acquired"
    )

    print(
        "  2. A sensor measurement that was unavailable"
    )

    print(
        "\nThe availability mask prevents inactive historical "
        "sensors from being treated as observed measurements."
    )

else:

    print(
        "CORRECTED ESTIMATOR INTERFACE: NOT VERIFIED"
    )

    print("=" * 70)
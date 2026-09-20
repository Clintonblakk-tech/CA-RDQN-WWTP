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
print("ESTIMATOR INFORMATION-AVAILABILITY AUDIT")
print("=" * 70)

print(f"\nCandidate sensors: {N_SENSORS}")
print(f"Feasible actions: {len(ACTIONS)}")
print(f"History length: {HISTORY}")


# ============================================================
# SYNTHETIC SENSOR HISTORY
#
# Every sensor receives a unique value at every time step.
# This makes accidental access to unavailable measurements
# immediately detectable.
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


print("\nComplete underlying sensor history:")
print(sensor_history)


# ============================================================
# HISTORICAL ACTIVATION HISTORY
#
# A different four-sensor configuration is active at each
# historical time step.
# ============================================================

historical_actions = [
    0,
    100,
    250,
    494,
]

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


print("\nHistorical activation masks:")

for t in range(HISTORY):

    selected_names = [
        SENSOR_NAMES[sensor_index]
        for sensor_index in range(N_SENSORS)
        if activation_history[t, sensor_index] == 1
    ]

    print(
        f"t-{HISTORY - t}: "
        f"{selected_names}"
    )


# ============================================================
# CURRENT ACTION
#
# Choose an action whose sensors were not all active
# throughout the historical window.
# ============================================================

current_action = 250

current_selected = ACTIONS[current_action]

print("\nCurrent action:")
print(f"Action ID: {current_action}")

print(
    "Selected sensors:",
    [
        SENSOR_NAMES[sensor_index]
        for sensor_index in current_selected
    ]
)


# ============================================================
# CURRENT PROTOTYPE BEHAVIOUR
#
# The prototype estimator receives the historical values
# of the four sensors belonging to the CURRENT action,
# regardless of whether those measurements were actually
# acquired at the historical time step.
#
# This is deliberately tested for information leakage.
# ============================================================

prototype_input = sensor_history[
    :,
    list(current_selected)
]

print("\nCURRENT PROTOTYPE ESTIMATOR INPUT")
print("----------------------------------")

print(prototype_input)


# ============================================================
# ACTUALLY AVAILABLE SENSOR HISTORY
#
# Apply the historical activation masks.
#
# NaN is used ONLY for this audit to make unavailable
# measurements visually obvious.
# ============================================================

available_history = sensor_history.copy().astype(float)

available_history[
    activation_history == 0
] = np.nan


print("\nACTUALLY AVAILABLE SENSOR HISTORY")
print("---------------------------------")

print(available_history)


# ============================================================
# LEGITIMATE INPUT FOR CURRENT ACTION
#
# Keep a historical value only when that sensor was actually
# active at that historical time step.
# ============================================================

legitimate_input = np.full(
    (HISTORY, len(current_selected)),
    np.nan,
    dtype=float
)

for t in range(HISTORY):

    for k, sensor_index in enumerate(current_selected):

        if activation_history[t, sensor_index] == 1:

            legitimate_input[t, k] = (
                sensor_history[t, sensor_index]
            )


print("\nLEGITIMATE INPUT FOR CURRENT ACTION")
print("-----------------------------------")

print(legitimate_input)


# ============================================================
# INFORMATION-AVAILABILITY VIOLATION TEST
# ============================================================

violations = []

for t in range(HISTORY):

    for k, sensor_index in enumerate(current_selected):

        prototype_value = prototype_input[t, k]

        was_available = (
            activation_history[t, sensor_index] == 1
        )

        if not was_available:

            violations.append(
                {
                    "history_step": t,
                    "sensor_index": sensor_index,
                    "sensor_name": SENSOR_NAMES[sensor_index],
                    "value_accessed": prototype_value,
                }
            )


# ============================================================
# AUDIT RESULT
# ============================================================

print("\nINFORMATION-AVAILABILITY CHECK")
print("--------------------------------")

if violations:

    print(
        "CURRENT PROTOTYPE: "
        "INFORMATION VIOLATION DETECTED"
    )

    print(
        f"Unavailable measurements accessed: "
        f"{len(violations)}"
    )

    for violation in violations:

        print(
            f"  history step "
            f"{violation['history_step']}, "
            f"{violation['sensor_name']}, "
            f"underlying value = "
            f"{violation['value_accessed']}"
        )

else:

    print(
        "CURRENT PROTOTYPE: "
        "NO INFORMATION VIOLATION"
    )


# ============================================================
# FINAL AUDIT SUMMARY
# ============================================================

print("\n" + "=" * 70)

if violations:

    print(
        "AUDIT RESULT: "
        "PROTOTYPE ESTIMATOR USES UNAVAILABLE HISTORY"
    )

    print("=" * 70)

    print(
        "\nThis confirms that the current estimator "
        "interface can access historical sensor values "
        "even when those sensors were inactive at those "
        "historical time steps."
    )

    print(
        "\nThe environment therefore requires an "
        "estimator-interface correction before DQN "
        "training."
    )

else:

    print(
        "AUDIT RESULT: "
        "NO INFORMATION VIOLATION"
    )

    print("=" * 70)
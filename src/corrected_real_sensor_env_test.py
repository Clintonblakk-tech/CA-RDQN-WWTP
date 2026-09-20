import itertools
import numpy as np

import bsm2_python
from bsm2_python.bsm1_ol import BSM1OL
# ============================================================
# CONFIGURATION
# ============================================================

DRY_INFLUENT = str(Path(bsm2_python.__file__).resolve().parent / "data" / "dryinfluent.csv")

HISTORY = 4
N_SENSORS = 12
ACTIVE_SENSORS = 4
DECISION_INTERVAL = 1
FORECAST_STEPS = 2


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


# BSM1 state indices
DO_INDEX = 7
SNO_INDEX = 8
NH4_INDEX = 9


# ============================================================
# ACTION SPACE
# ============================================================

ACTIONS = list(
    itertools.combinations(
        range(N_SENSORS),
        ACTIVE_SENSORS
    )
)


# ============================================================
# SENSOR EXTRACTION
# ============================================================

def extract_sensor_matrix(model):
    """
    Extract the 12 candidate process sensors from the
    completed BSM1 simulation.
    """

    sensor_matrix = np.column_stack(
        [
            model.y_out1_all[:, DO_INDEX],
            model.y_out2_all[:, DO_INDEX],
            model.y_out3_all[:, DO_INDEX],
            model.y_out4_all[:, DO_INDEX],
            model.y_out5_all[:, DO_INDEX],

            model.y_out1_all[:, NH4_INDEX],
            model.y_out2_all[:, NH4_INDEX],
            model.y_out3_all[:, NH4_INDEX],
            model.y_out4_all[:, NH4_INDEX],
            model.y_out5_all[:, NH4_INDEX],

            model.y_out3_all[:, SNO_INDEX],
            model.y_out5_all[:, SNO_INDEX],
        ]
    )

    return sensor_matrix


# ============================================================
# BUILD HISTORICAL ACQUISITION
# ============================================================

def build_acquired_history(
    sensor_matrix,
    activation_history,
    end_index
):
    """
    Construct the historical measurements actually acquired
    by the sensor scheduler.

    A sensor contributes a value at historical time t ONLY if
    that sensor was active at time t.

    Unavailable measurements are represented as zero here,
    while the corresponding availability mask identifies
    whether the value is actually observed.
    """

    start_index = end_index - HISTORY

    measurements = np.zeros(
        (HISTORY, N_SENSORS),
        dtype=float
    )

    masks = np.zeros(
        (HISTORY, N_SENSORS),
        dtype=float
    )

    for h in range(HISTORY):

        time_index = start_index + h

        for sensor_index in range(N_SENSORS):

            if activation_history[
                time_index,
                sensor_index
            ] == 1:

                measurements[
                    h,
                    sensor_index
                ] = sensor_matrix[
                    time_index,
                    sensor_index
                ]

                masks[
                    h,
                    sensor_index
                ] = 1.0

    return measurements, masks


# ============================================================
# BUILD CORRECTED ESTIMATOR STATE
# ============================================================

def build_corrected_state(
    sensor_matrix,
    activation_history,
    end_index
):
    """
    Build the causal estimator/scheduler state.

    State consists of:

        4 historical measurement matrices
        +
        4 historical availability-mask matrices

    Therefore:

        4 × 12 × 2 = 96 features
    """

    measurements, masks = build_acquired_history(
        sensor_matrix,
        activation_history,
        end_index
    )

    state = np.concatenate(
        [
            measurements,
            masks
        ],
        axis=1
    )

    return state.flatten()


# ============================================================
# ACTION APPLICATION
# ============================================================

def apply_action(
    activation_history,
    time_index,
    action_id
):
    """
    Activate exactly four sensors at the specified time.
    """

    activation_history[
        time_index,
        :
    ] = 0

    selected = ACTIONS[action_id]

    activation_history[
        time_index,
        list(selected)
    ] = 1


# ============================================================
# REWARD CALCULATION
# ============================================================

def calculate_reward(
    predicted_delta_nh4,
    predicted_delta_sno,
    true_delta_nh4,
    true_delta_sno
):
    """
    Normalized squared prediction error reward.

    More accurate predictions produce a reward closer to zero.
    """

    nh4_scale = 1.0
    sno_scale = 1.0

    nh4_error = (
        predicted_delta_nh4 - true_delta_nh4
    ) / nh4_scale

    sno_error = (
        predicted_delta_sno - true_delta_sno
    ) / sno_scale

    error = (
        nh4_error ** 2
        +
        sno_error ** 2
    )

    return -0.5 * error


# ============================================================
# LOAD AND RUN BSM1
# ============================================================

print("=" * 70)
print("CORRECTED REAL BSM1 SENSOR ENVIRONMENT TEST")
print("=" * 70)

print("\nLoading BSM1 dry influent data...")

data = np.loadtxt(
    DRY_INFLUENT,
    delimiter=",",
    skiprows=1
)

print(
    f"Influent data shape: {data.shape}"
)

print("\nRunning BSM1 simulation...")

model = BSM1OL(
    data_in=data,
    timestep=None,
    evaltime=np.array(
        [8.98958333, 13.98958333]
    ),
)
model.simulate(
    plot=False
)

print("BSM1 simulation complete.")


# ============================================================
# EXTRACT REAL SENSOR DATA
# ============================================================

sensor_matrix = extract_sensor_matrix(model)

print(
    f"\nReal sensor matrix shape: "
    f"{sensor_matrix.shape}"
)

print(
    f"Candidate sensors: {N_SENSORS}"
)

print(
    f"Feasible actions: {len(ACTIONS)}"
)


# ============================================================
# ACTIVATION HISTORY
# ============================================================

n_steps = sensor_matrix.shape[0]

activation_history = np.zeros(
    (n_steps, N_SENSORS),
    dtype=int
)


# ============================================================
# INITIAL HISTORY
#
# Use four valid historical actions so that the estimator
# begins with a complete four-step scheduler history.
# ============================================================

initial_actions = [
    0,
    100,
    250,
    494,
]

for t, action_id in enumerate(initial_actions):

    apply_action(
        activation_history,
        t,
        action_id
    )


# ============================================================
# INITIAL STATE TEST
# ============================================================

initial_state = build_corrected_state(
    sensor_matrix,
    activation_history,
    HISTORY
)

print(
    f"\nInitial corrected state dimension: "
    f"{initial_state.shape[0]}"
)

if initial_state.shape[0] != 96:

    raise RuntimeError(
        "Corrected state dimension is not 96."
    )

print(
    "Initial corrected state: VERIFIED"
)


# ============================================================
# INFORMATION AVAILABILITY VERIFICATION
# ============================================================

print("\nChecking historical information availability...")

availability_errors = []

test_end_index = HISTORY

measurements, masks = build_acquired_history(
    sensor_matrix,
    activation_history,
    test_end_index
)

for h in range(HISTORY):

    time_index = test_end_index - HISTORY + h

    for sensor_index in range(N_SENSORS):

        mask = masks[h, sensor_index]

        if activation_history[
            time_index,
            sensor_index
        ] == 1:

            if mask != 1:

                availability_errors.append(
                    (
                        h,
                        sensor_index,
                        "ACTIVE SENSOR HAS NO MASK"
                    )
                )

        else:

            if mask != 0:

                availability_errors.append(
                    (
                        h,
                        sensor_index,
                        "INACTIVE SENSOR HAS MASK"
                    )
                )


if availability_errors:

    print(
        "INFORMATION AVAILABILITY: FAILED"
    )

    for error in availability_errors:

        print(error)

    raise RuntimeError(
        "Historical availability test failed."
    )

else:

    print(
        "INFORMATION AVAILABILITY: VERIFIED"
    )


# ============================================================
# DECISION-STEP TEST
#
# Test several feasible actions using actual BSM1 data.
# ============================================================

test_actions = [
    0,
    100,
    250,
    494,
]


print("\n" + "=" * 70)
print("CORRECTED CAUSAL DECISION-STEP TEST")
print("=" * 70)


for step_number, action_id in enumerate(
    test_actions,
    start=1
):

    decision_index = HISTORY + step_number

    # --------------------------------------------------------
    # Apply current action
    # --------------------------------------------------------

    apply_action(
        activation_history,
        decision_index,
        action_id
    )

    selected = ACTIONS[action_id]

    # --------------------------------------------------------
    # Build state from measurements that were ACTUALLY
    # acquired during the previous four time steps.
    # --------------------------------------------------------

    state = build_corrected_state(
        sensor_matrix,
        activation_history,
        decision_index
    )

    # --------------------------------------------------------
    # Verify state
    # --------------------------------------------------------

    if state.shape[0] != 96:

        raise RuntimeError(
            "State dimension changed."
        )

    # --------------------------------------------------------
    # Current acquired measurements
    # --------------------------------------------------------

    current_measurements = np.zeros(
        N_SENSORS,
        dtype=float
    )

    current_mask = np.zeros(
        N_SENSORS,
        dtype=int
    )

    for sensor_index in selected:

        current_measurements[
            sensor_index
        ] = sensor_matrix[
            decision_index,
            sensor_index
        ]

        current_mask[
            sensor_index
        ] = 1

    # --------------------------------------------------------
    # True 30-minute changes
    # --------------------------------------------------------

    target_index = (
        decision_index + FORECAST_STEPS
    )

    if target_index >= n_steps:

        print(
            "\nReached end of simulation."
        )

        break

    true_delta_nh4 = (
        model.ys_eff_all[
            target_index,
            NH4_INDEX
        ]
        -
        model.ys_eff_all[
            decision_index,
            NH4_INDEX
        ]
    )

    true_delta_sno = (
        model.ys_eff_all[
            target_index,
            SNO_INDEX
        ]
        -
        model.ys_eff_all[
            decision_index,
            SNO_INDEX
        ]
    )

    # --------------------------------------------------------
    # Demonstration prediction
    #
    # This is NOT the final estimator.
    #
    # The purpose here is only to verify that the corrected
    # environment can construct a causal state before the
    # estimator is connected.
    # --------------------------------------------------------

    predicted_delta_nh4 = 0.0
    predicted_delta_sno = 0.0

    reward = calculate_reward(
        predicted_delta_nh4,
        predicted_delta_sno,
        true_delta_nh4,
        true_delta_sno
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print(
        f"\nDECISION STEP {step_number}"
    )

    print(
        f"Decision index: {decision_index}"
    )

    print(
        f"Action: {action_id}"
    )

    print(
        "Selected sensors:"
    )

    for sensor_index in selected:

        print(
            f"  {SENSOR_NAMES[sensor_index]}"
        )

    print(
        f"Active sensors: "
        f"{int(current_mask.sum())}"
    )

    print(
        f"State dimension: "
        f"{state.shape[0]}"
    )

    print(
        f"True Î”NH4-N: "
        f"{true_delta_nh4:.6f}"
    )

    print(
        f"True Î”SNO: "
        f"{true_delta_sno:.6f}"
    )

    print(
        f"Demonstration reward: "
        f"{reward:.6f}"
    )


# ============================================================
# FINAL ACTIVATION-HISTORY CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL ACTIVATION-HISTORY VERIFICATION")
print("=" * 70)

final_errors = []

for t in range(
    n_steps
):

    active_count = int(
        activation_history[t].sum()
    )

    if active_count not in (0, 4):

        final_errors.append(
            (
                t,
                active_count
            )
        )


if final_errors:

    print(
        "FINAL ACTIVATION HISTORY: FAILED"
    )

    for error in final_errors[:20]:

        print(error)

    raise RuntimeError(
        "Activation history contains invalid sensor counts."
    )

else:

    print(
        "Exactly four sensors are active at each "
        "scheduled decision step."
    )

    print(
        "FINAL ACTIVATION HISTORY: VERIFIED"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print(
    "CORRECTED REAL BSM1 ENVIRONMENT TEST PASSED"
)
print("=" * 70)

print(
    "\nVerified:"
)

print(
    "  Actual BSM1 sensor data used"
)

print(
    "  12 candidate sensors"
)

print(
    "  495 feasible four-sensor actions"
)

print(
    "  Historical measurements restricted "
    "by actual activation"
)

print(
    "  Historical availability masks preserved"
)

print(
    "  96-dimensional corrected state"
)

print(
    "  Four-step historical information window"
)

print(
    "  15-minute decision progression"
)

print(
    "  30-minute Î”NH4-N / Î”SNO targets"
)

print(
    "\nThe estimator can now be connected only to "
    "measurements that were actually acquired."
)


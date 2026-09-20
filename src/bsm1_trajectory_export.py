import os
import numpy as np

import bsm2_python
from bsm2_python.bsm1_ol import BSM1OL


# ============================================================
# STEP 50 — BSM1 TRAJECTORY EXPORT
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DRY_INFLUENT = str(Path(bsm2_python.__file__).resolve().parent / "data" / "dryinfluent.csv")

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "bsm1"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1. Load validated BSM1 dry influent
# ------------------------------------------------------------

data = np.loadtxt(
    DRY_INFLUENT,
    delimiter=",",
    skiprows=1
)

print("=" * 70)
print("STEP 50 — BSM1 TRAJECTORY EXPORT")
print("=" * 70)

print(f"[INFO] Influent shape: {data.shape}")


# ------------------------------------------------------------
# 2. Run BSM1
# ------------------------------------------------------------

model = BSM1OL(
    data_in=data,
    timestep=None,
    evaltime=np.array([8.98958333, 13.98958333])
)

model.simulate(plot=False)

print("[PASS] BSM1 simulation completed.")


# ------------------------------------------------------------
# 3. Extract validated trajectories
# ------------------------------------------------------------

reactor_outputs = np.column_stack(
    [
        model.y_out1_all,
        model.y_out2_all,
        model.y_out3_all,
        model.y_out4_all,
        model.y_out5_all,
    ]
)

effluent = model.ys_eff_all


# ------------------------------------------------------------
# 4. Validate dimensions
# ------------------------------------------------------------

assert reactor_outputs.ndim == 2
assert effluent.ndim == 2

assert reactor_outputs.shape[0] == effluent.shape[0]
assert reactor_outputs.shape[1] == 21 * 5
assert effluent.shape[1] == 21

assert np.isfinite(reactor_outputs).all()
assert np.isfinite(effluent).all()

print(
    f"[PASS] Reactor trajectory shape: "
    f"{reactor_outputs.shape}"
)

print(
    f"[PASS] Effluent trajectory shape: "
    f"{effluent.shape}"
)


# ------------------------------------------------------------
# 5. Save trajectories
# ------------------------------------------------------------

np.save(
    os.path.join(
        OUTPUT_DIR,
        "bsm1_reactor_trajectories.npy"
    ),
    reactor_outputs
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "bsm1_effluent_trajectories.npy"
    ),
    effluent
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "bsm1_time.npy"
    ),
    model.simtime
)


# ------------------------------------------------------------
# 6. Save metadata
# ------------------------------------------------------------

sensor_names = [
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

target_names = [
    "Effluent_NH4",
    "Effluent_SNO",
    "Effluent_TSS",
]

with open(
    os.path.join(
        OUTPUT_DIR,
        "bsm1_trajectory_metadata.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write("BSM1 trajectory export\n")
    f.write("=====================\n")
    f.write("Model: BSM1OL\n")
    f.write("Influent: dryinfluent.csv\n")
    f.write("Solver timestep: native BSM1 input timestep\n")
    f.write("Decision interval: 15 minutes\n")
    f.write("Forecast horizon: 30 minutes\n")
    f.write("\n")

    f.write("Candidate sensors:\n")
    for name in sensor_names:
        f.write(f"- {name}\n")

    f.write("\nTargets:\n")
    for name in target_names:
        f.write(f"- {name}\n")


# ------------------------------------------------------------
# 7. Final report
# ------------------------------------------------------------

print("[PASS] Reactor trajectories saved.")
print("[PASS] Effluent trajectories saved.")
print("[PASS] Time vector saved.")
print("[PASS] Metadata saved.")

print()
print(f"[INFO] Output directory:")
print(OUTPUT_DIR)

print("=" * 70)
print("STEP 50 COMPLETED")
print("=" * 70)
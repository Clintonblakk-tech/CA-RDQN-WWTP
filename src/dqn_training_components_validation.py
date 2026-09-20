from itertools import combinations
from collections import deque

import numpy as np


# ============================================================
# STEP 24 — DQN TRAINING COMPONENTS VALIDATION
# ============================================================

STATE_DIM = 96
ACTION_DIM = 495
N_SENSORS = 12
N_ACTIVE = 4
HISTORY = 4

GAMMA = 0.99

EPSILON_START = 1.00
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 20_000

TARGET_UPDATE_FREQUENCY = 500

REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64

TRAIN_END = 863
EVAL_START = 863

RANDOM_SEED = 20260916


print("\n" + "=" * 70)
print("STEP 24 — DQN TRAINING COMPONENTS VALIDATION")
print("=" * 70)


# ============================================================
# 1. Feasible action space
# ============================================================

actions = list(
    combinations(
        range(N_SENSORS),
        N_ACTIVE,
    )
)

assert len(actions) == ACTION_DIM

for action in actions:

    assert len(action) == N_ACTIVE
    assert len(set(action)) == N_ACTIVE

print("\n1. Action-space validation")
print(f"  Number of actions       : {len(actions)}")
print(f"  Sensors per action     : {N_ACTIVE}")
print("  [PASS] All actions are feasible.")


# ============================================================
# 2. Replay buffer
# ============================================================

class ReplayBuffer:

    def __init__(
        self,
        capacity,
    ):

        self.buffer = deque(
            maxlen=capacity
        )

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.buffer.append(
            (
                np.asarray(
                    state,
                    dtype=np.float32,
                ),
                int(action),
                float(reward),
                (
                    None
                    if next_state is None
                    else np.asarray(
                        next_state,
                        dtype=np.float32,
                    )
                ),
                bool(done),
            )
        )

    def sample(
        self,
        batch_size,
        rng,
    ):

        if len(self.buffer) < batch_size:
            raise ValueError(
                "Not enough transitions."
            )

        indices = rng.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
        )

        batch = [
            self.buffer[i]
            for i in indices
        ]

        states = np.stack(
            [x[0] for x in batch]
        )

        actions = np.asarray(
            [x[1] for x in batch],
            dtype=np.int64,
        )

        rewards = np.asarray(
            [x[2] for x in batch],
            dtype=np.float32,
        )

        next_states = [
            x[3]
            for x in batch
        ]

        dones = np.asarray(
            [x[4] for x in batch],
            dtype=np.float32,
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

    def __len__(self):

        return len(self.buffer)


# ============================================================
# 3. Replay-buffer validation
# ============================================================

print("\n2. Replay-buffer validation")

rng = np.random.default_rng(
    RANDOM_SEED
)

buffer = ReplayBuffer(
    REPLAY_CAPACITY
)

for i in range(100):

    state = rng.normal(
        0.0,
        1.0,
        STATE_DIM,
    )

    next_state = rng.normal(
        0.0,
        1.0,
        STATE_DIM,
    )

    action = i % ACTION_DIM

    reward = -float(
        rng.random()
    )

    done = (
        i == 99
    )

    buffer.add(
        state,
        action,
        reward,
        next_state,
        done,
    )

assert len(buffer) == 100

(
    states,
    sampled_actions,
    rewards,
    next_states,
    dones,
) = buffer.sample(
    BATCH_SIZE,
    rng,
)

assert states.shape == (
    BATCH_SIZE,
    STATE_DIM,
)

assert sampled_actions.shape == (
    BATCH_SIZE,
)

assert rewards.shape == (
    BATCH_SIZE,
)

assert len(next_states) == BATCH_SIZE

assert dones.shape == (
    BATCH_SIZE,
)

assert np.all(
    (sampled_actions >= 0)
    &
    (sampled_actions < ACTION_DIM)
)

print(
    f"  Buffer size             : {len(buffer)}"
)

print(
    f"  Sample batch size       : {BATCH_SIZE}"
)

print(
    f"  State batch shape       : {states.shape}"
)

print("  [PASS] Replay buffer works.")


# ============================================================
# 4. Epsilon-greedy schedule
# ============================================================

def epsilon_by_step(step):

    fraction = min(
        step / EPSILON_DECAY_STEPS,
        1.0,
    )

    return (
        EPSILON_START
        +
        fraction
        * (
            EPSILON_END
            - EPSILON_START
        )
    )


print("\n3. Epsilon schedule")

epsilon_values = [
    epsilon_by_step(0),
    epsilon_by_step(5_000),
    epsilon_by_step(10_000),
    epsilon_by_step(20_000),
    epsilon_by_step(40_000),
]

for step, epsilon in zip(
    [0, 5_000, 10_000, 20_000, 40_000],
    epsilon_values,
):

    print(
        f"  Step {step:>6}: "
        f"epsilon = {epsilon:.4f}"
    )

assert np.isclose(
    epsilon_by_step(0),
    1.0,
)

assert np.isclose(
    epsilon_by_step(
        EPSILON_DECAY_STEPS
    ),
    EPSILON_END,
)

assert np.isclose(
    epsilon_by_step(
        EPSILON_DECAY_STEPS * 2
    ),
    EPSILON_END,
)

print("  [PASS] Epsilon schedule is bounded.")


# ============================================================
# 5. Epsilon-greedy action selection
# ============================================================

def select_action(
    q_values,
    epsilon,
    rng,
):

    if rng.random() < epsilon:

        return int(
            rng.integers(
                0,
                ACTION_DIM,
            )
        )

    return int(
        np.argmax(
            q_values
        )
    )


print("\n4. Epsilon-greedy action selection")

test_q_values = np.linspace(
    0.0,
    1.0,
    ACTION_DIM,
)

greedy_action = select_action(
    test_q_values,
    epsilon=0.0,
    rng=rng,
)

assert greedy_action == (
    ACTION_DIM - 1
)

print(
    f"  Greedy action index     : "
    f"{greedy_action}"
)

random_actions = [
    select_action(
        test_q_values,
        epsilon=1.0,
        rng=rng,
    )
    for _ in range(100)
]

assert all(
    0 <= a < ACTION_DIM
    for a in random_actions
)

print(
    f"  Random actions tested   : "
    f"{len(random_actions)}"
)

print(
    f"  Unique random actions  : "
    f"{len(set(random_actions))}"
)

print("  [PASS] Epsilon-greedy selection works.")


# ============================================================
# 6. DQN target calculation
# ============================================================

print("\n5. Bellman target validation")

test_rewards = np.array(
    [
        -0.2,
        -1.0,
        -2.0,
        -0.5,
    ],
    dtype=np.float32,
)

test_next_q = np.array(
    [
        0.4,
        0.8,
        1.2,
        0.0,
    ],
    dtype=np.float32,
)

test_dones = np.array(
    [
        0.0,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float32,
)

bellman_targets = (
    test_rewards
    +
    GAMMA
    * (
        1.0 - test_dones
    )
    * test_next_q
)

expected_targets = np.array(
    [
        -0.2 + 0.99 * 0.4,
        -1.0 + 0.99 * 0.8,
        -2.0 + 0.99 * 1.2,
        -0.5,
    ],
    dtype=np.float32,
)

assert np.allclose(
    bellman_targets,
    expected_targets,
)

print(
    "  Terminal transition "
    "correctly excludes future Q-value."
)

print(
    "  Non-terminal transitions "
    "include discounted future Q-value."
)

print("  [PASS] Bellman targets validated.")


# ============================================================
# 7. Target-network update schedule
# ============================================================

def should_update_target(
    training_step
):

    return (
        training_step > 0
        and
        training_step
        % TARGET_UPDATE_FREQUENCY
        == 0
    )


print("\n6. Target-network update schedule")

test_steps = [
    0,
    499,
    500,
    999,
    1000,
    1500,
]

for step in test_steps:

    print(
        f"  Step {step:>4}: "
        f"update = "
        f"{should_update_target(step)}"
    )

assert not should_update_target(0)
assert not should_update_target(499)
assert should_update_target(500)
assert should_update_target(1000)
assert should_update_target(1500)

print(
    "  [PASS] Target-network schedule validated."
)


# ============================================================
# 8. Chronological split validation
# ============================================================

print("\n7. Chronological split")

assert TRAIN_END == EVAL_START

train_indices = np.arange(
    HISTORY,
    TRAIN_END,
)

eval_indices = np.arange(
    EVAL_START,
    1343 - 2,
)

assert train_indices.max() < eval_indices.min()

print(
    f"  Training indices: "
    f"{train_indices.min()}–"
    f"{train_indices.max()}"
)

print(
    f"  Evaluation indices: "
    f"{eval_indices.min()}–"
    f"{eval_indices.max()}"
)

print(
    "  [PASS] Training and evaluation periods "
    "are chronologically separated."
)


# ============================================================
# 9. Seed reproducibility
# ============================================================

print("\n8. Random-seed reproducibility")

rng_a = np.random.default_rng(
    RANDOM_SEED
)

rng_b = np.random.default_rng(
    RANDOM_SEED
)

sequence_a = rng_a.integers(
    0,
    ACTION_DIM,
    size=100,
)

sequence_b = rng_b.integers(
    0,
    ACTION_DIM,
    size=100,
)

assert np.array_equal(
    sequence_a,
    sequence_b,
)

print(
    "  [PASS] Identical seeds produce "
    "identical action sequences."
)


# ============================================================
# 10. Different-seed independence
# ============================================================

rng_c = np.random.default_rng(
    RANDOM_SEED + 1
)

sequence_c = rng_c.integers(
    0,
    ACTION_DIM,
    size=100,
)

assert not np.array_equal(
    sequence_a,
    sequence_c,
)

print(
    "  [PASS] Different seeds produce "
    "different action sequences."
)


# ============================================================
# 11. Training-step budget
# ============================================================

print("\n9. Training configuration")

print(
    f"  Replay capacity         : "
    f"{REPLAY_CAPACITY}"
)

print(
    f"  Batch size              : "
    f"{BATCH_SIZE}"
)

print(
    f"  Gamma                   : "
    f"{GAMMA}"
)

print(
    f"  Epsilon start           : "
    f"{EPSILON_START}"
)

print(
    f"  Epsilon end             : "
    f"{EPSILON_END}"
)

print(
    f"  Epsilon decay steps     : "
    f"{EPSILON_DECAY_STEPS}"
)

print(
    f"  Target update frequency : "
    f"{TARGET_UPDATE_FREQUENCY}"
)

assert REPLAY_CAPACITY > BATCH_SIZE
assert 0.0 < GAMMA <= 1.0
assert 0.0 <= EPSILON_END <= EPSILON_START <= 1.0

print(
    "  [PASS] Training configuration "
    "is internally consistent."
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("STEP 24 RESULTS")
print("=" * 70)

print("[PASS] 495-action constrained action space.")
print("[PASS] Replay buffer.")
print("[PASS] 64-transition mini-batch.")
print("[PASS] Epsilon-greedy exploration.")
print("[PASS] Bellman target calculation.")
print("[PASS] Target-network update schedule.")
print("[PASS] Chronological train/evaluation separation.")
print("[PASS] Reproducible random seeds.")
print("[PASS] Independent random seeds.")
print("[PASS] Training configuration consistency.")

print("\nProposed DQN configuration:")
print("  State dimension     = 96")
print("  Action dimension    = 495")
print("  Hidden layers       = 128 → 64")
print("  Gamma               = 0.99")
print("  Batch size          = 64")
print("  Replay capacity     = 50,000")
print("  Epsilon             = 1.00 → 0.05")
print("  Epsilon decay       = 20,000 steps")
print("  Target update       = every 500 steps")

print("\n" + "=" * 70)
print("STEP 24 COMPLETED")
print("=" * 70)
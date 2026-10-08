import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
# PSO-DWA MOBILE ROBOT SIMULATION
# Simplified TurtleBot3-like differential drive robot
# ============================================================

np.random.seed(2)

# -----------------------------
# Environment
# -----------------------------

obstacles = np.array([
    [4, 4],
    [6, 6],
    [4, 7],
    [7, 3],
    [8, 7]
])

start = np.array([1.0, 1.0, 0.0])
goal = np.array([10.0, 10.0])

robot_radius = 0.35

# -----------------------------
# Robot parameters
# -----------------------------

DT = 0.25

MAX_SPEED = 1.0
MAX_OMEGA = 1.2

# ============================================================
# Utility functions
# ============================================================

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ============================================================
# Collision checking
# ============================================================

def collision(x, y):

    # Boundary
    if x < 0.3 or x > 11.5 or y < 0.3 or y > 11.5:
        return True

    robot_position = np.array([x, y])

    distances = np.linalg.norm(
        obstacles - robot_position,
        axis=1
    )

    return np.any(distances <= robot_radius)


# ============================================================
# Simulate robot motion for a short time
# ============================================================

def simulate_trajectory(state, velocity, omega, horizon=1.5):

    x, y, theta = state

    points = []

    steps = int(horizon / DT)

    for _ in range(steps):

        x = x + velocity * np.cos(theta) * DT
        y = y + velocity * np.sin(theta) * DT

        theta = normalize_angle(
            theta + omega * DT
        )

        points.append([x, y])

        if collision(x, y):

            return (
                np.array(points),
                True,
                np.array([x, y, theta])
            )

    return (
        np.array(points),
        False,
        np.array([x, y, theta])
    )


# ============================================================
# Calculate trajectory features
# ============================================================

def calculate_features(state, velocity, omega):

    trajectory, collision_flag, final_state = \
        simulate_trajectory(
            state,
            velocity,
            omega
        )

    if len(trajectory) == 0:
        return None

    current_distance = np.linalg.norm(
        goal - state[:2]
    )

    final_distance = np.linalg.norm(
        goal - final_state[:2]
    )

    # Progress toward goal
    progress = max(
        0,
        current_distance - final_distance
    )

    # Distance from obstacles
    obstacle_distances = np.linalg.norm(
        trajectory[:, None, :] -
        obstacles[None, :, :],
        axis=2
    )

    clearance = np.min(obstacle_distances) - robot_radius

    clearance = max(clearance, 0)

    # Direction toward goal
    direction = np.array([
        np.cos(final_state[2]),
        np.sin(final_state[2])
    ])

    goal_vector = goal - final_state[:2]

    if np.linalg.norm(goal_vector) > 0:

        alignment = np.dot(
            direction,
            goal_vector
        ) / np.linalg.norm(goal_vector)

    else:

        alignment = 1

    alignment = (alignment + 1) / 2

    return (
        progress,
        clearance,
        velocity,
        alignment,
        collision_flag,
        final_state
    )


# ============================================================
# Generate possible DWA velocities
# ============================================================

def generate_candidates(state):

    candidates = []

    velocities = np.linspace(
        0,
        MAX_SPEED,
        6
    )

    angular_velocities = np.linspace(
        -MAX_OMEGA,
        MAX_OMEGA,
        9
    )

    for v in velocities:

        for omega in angular_velocities:

            features = calculate_features(
                state,
                v,
                omega
            )

            if features is not None:

                # Ignore trajectories that collide
                if not features[4]:

                    candidates.append(
                        (v, omega, features)
                    )

    return candidates


# ============================================================
# Normalize DWA features
# ============================================================

def normalize_features(candidates):

    data = np.array([
        [
            c[2][0],   # progress
            c[2][1],   # clearance
            c[2][2]    # velocity
        ]

        for c in candidates
    ])

    minimum = data.min(axis=0)
    maximum = data.max(axis=0)

    normalized = (
        data - minimum
    ) / (
        maximum - minimum + 1e-9
    )

    return normalized


# ============================================================
# Select best DWA movement
# ============================================================

def best_movement(candidates, weights):

    normalized = normalize_features(
        candidates
    )

    scores = normalized @ weights

    best_index = np.argmax(scores)

    return (
        best_index,
        scores[best_index]
    )


# ============================================================
# PARTICLE SWARM OPTIMIZATION
# ============================================================

def PSO(candidates):

    particles = 18
    iterations = 12

    # Random particles
    positions = np.random.uniform(
        0.1,
        1.0,
        (particles, 3)
    )

    velocities = np.random.normal(
        0,
        0.08,
        (particles, 3)
    )

    # Personal best
    personal_best = positions.copy()

    personal_score = np.zeros(
        particles
    )

    # Fitness function
    def fitness(weights):

        weights = weights / np.sum(weights)

        index, score = best_movement(
            candidates,
            weights
        )

        features = candidates[index][2]

        progress = features[0]
        clearance = features[1]
        speed = features[2]
        alignment = features[3]

        fitness_value = (
            2.5 * progress
            + 0.8 * clearance
            + 0.5 * speed
            + 0.7 * alignment
        )

        return fitness_value

    # Initial fitness
    for i in range(particles):

        personal_score[i] = fitness(
            positions[i]
        )

    # Global best
    best_particle = np.argmax(
        personal_score
    )

    global_best = personal_best[
        best_particle
    ].copy()

    global_score = personal_score[
        best_particle
    ]

    # PSO iterations
    for iteration in range(iterations):

        r1 = np.random.random(
            (particles, 3)
        )

        r2 = np.random.random(
            (particles, 3)
        )

        # Velocity equation
        velocities = (
            0.7 * velocities
            + 1.6 * r1 *
            (personal_best - positions)
            + 1.8 * r2 *
            (global_best - positions)
        )

        # Position equation
        positions = positions + velocities

        positions = np.clip(
            positions,
            0.01,
            2.0
        )

        # Evaluate particles
        for i in range(particles):

            score = fitness(
                positions[i]
            )

            if score > personal_score[i]:

                personal_score[i] = score

                personal_best[i] = \
                    positions[i].copy()

        # Update global best
        best_particle = np.argmax(
            personal_score
        )

        if personal_score[
            best_particle
        ] > global_score:

            global_score = \
                personal_score[best_particle]

            global_best = \
                personal_best[
                    best_particle
                ].copy()

    # Normalize final weights
    global_best = (
        global_best /
        np.sum(global_best)
    )

    return global_best


# ============================================================
# MAIN ROBOT SIMULATION
# ============================================================

state = start.copy()

path = [state.copy()]

print("\n===================================")
print("      PSO-DWA ROBOT SIMULATION")
print("===================================")

print("Start position :", start[:2])
print("Goal position  :", goal)

for step in range(150):

    # Generate possible movements
    candidates = generate_candidates(
        state
    )

    if len(candidates) == 0:

        print("Robot is stuck!")

        break

    # PSO finds optimal DWA weights
    weights = PSO(
        candidates
    )

    # DWA chooses best movement
    best_index, score = best_movement(
        candidates,
        weights
    )

    velocity, omega, features = \
        candidates[best_index]

    # Move robot for one time step
    trajectory, collision_flag, new_state = \
        simulate_trajectory(
            state,
            velocity,
            omega,
            horizon=DT
        )

    state = new_state

    path.append(
        state.copy()
    )

    # Check goal
    distance_to_goal = np.linalg.norm(
        goal - state[:2]
    )

    if distance_to_goal < 0.45:

        print("\nGoal reached!")

        break


# ============================================================
# RESULTS
# ============================================================

path = np.array(path)

travel_distance = np.sum(
    np.linalg.norm(
        np.diff(
            path[:, :2],
            axis=0
        ),
        axis=1
    )
)

final_distance = np.linalg.norm(
    goal - state[:2]
)

print("\n------------- RESULTS -------------")

print(
    "Number of steps :",
    len(path) - 1
)

print(
    "Travel distance :",
    round(travel_distance, 2)
)

print(
    "Final position  :",
    np.round(state[:2], 2)
)

print(
    "Distance to goal:",
    round(final_distance, 2)
)

print(
    "PSO weights     :",
    np.round(weights, 3)
)

print("-----------------------------------")


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(9, 7))

# Robot path
plt.plot(
    path[:, 0],
    path[:, 1],
    marker='o',
    markersize=2,
    label="Robot Path"
)

# Obstacles
plt.scatter(
    obstacles[:, 0],
    obstacles[:, 1],
    s=200,
    marker='s',
    label="Obstacles"
)

# Start
plt.scatter(
    start[0],
    start[1],
    s=150,
    marker='o',
    label="Start"
)

# Goal
plt.scatter(
    goal[0],
    goal[1],
    s=150,
    marker='*',
    label="Goal"
)

plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.title(
    "PSO-DWA Mobile Robot Path Planning"
)

plt.xlim(0, 12)
plt.ylim(0, 12)

plt.grid(True)
plt.legend()

plt.show()
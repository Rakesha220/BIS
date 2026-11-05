
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Environment settings
START = np.array([1.0, 1.0])
GOAL = np.array([9.0, 9.0])

OBSTACLES = [
    (3, 3, 1.0),
    (5, 5, 1.2),
    (7, 7, 1.0),
    (3, 7, 0.8)
]

# GWO parameters
WOLVES = 20
ITERATIONS = 100
WAYPOINTS = 5
DIM = WAYPOINTS * 2

LOWER = 0
UPPER = 10
CLEARANCE = 0.3


# Calculate distance between two points
def distance(p1, p2):
    return np.linalg.norm(p1 - p2)


# Calculate minimum distance from a circle
# to a line segment
def segment_distance(a, b, center):
    ab = b - a
    denom = np.dot(ab, ab)

    if denom == 0:
        return distance(a, center)

    t = np.dot(center - a, ab) / denom
    t = np.clip(t, 0, 1)
    closest = a + t * ab

    return distance(closest, center)


# Fitness function
def fitness(wolf):
    points = wolf.reshape(WAYPOINTS, 2)
    path = np.vstack([START, points, GOAL])

    # Total path length
    cost = sum(
        distance(path[i], path[i + 1])
        for i in range(len(path) - 1)
    )

    # Obstacle penalty
    for i in range(len(path) - 1):
        for ox, oy, radius in OBSTACLES:
            center = np.array([ox, oy])
            d = segment_distance(path[i], path[i + 1], center)
            safe_distance = radius + CLEARANCE

            if d < safe_distance:
                cost += 10000 * (safe_distance - d) ** 2

    return cost


# Initialize wolf population
def initialize_wolves():
    return np.random.uniform(
        LOWER, UPPER, (WOLVES, DIM)
    )


# Find Alpha, Beta and Delta wolves
def find_best_wolves(wolves, scores):
    order = np.argsort(scores)

    return (
        wolves[order[0]].copy(),
        wolves[order[1]].copy(),
        wolves[order[2]].copy()
    )


# Grey Wolf Optimization
def gwo():
    wolves = initialize_wolves()
    convergence = []

    for iteration in range(ITERATIONS):

        scores = np.array([
            fitness(wolf) for wolf in wolves
        ])

        alpha, beta, delta = find_best_wolves(
            wolves, scores
        )

        best_score = fitness(alpha)
        convergence.append(best_score)

        # Decrease a from 2 to 0
        a = 2 - 2 * iteration / ITERATIONS

        new_wolves = []

        for wolf in wolves:
            updated_positions = []

            for leader in [alpha, beta, delta]:

                r1 = np.random.rand(DIM)
                r2 = np.random.rand(DIM)

                A = 2 * a * r1 - a
                C = 2 * r2

                D = np.abs(C * leader - wolf)
                X = leader - A * D

                updated_positions.append(X)

            new_position = np.mean(
                updated_positions, axis=0
            )

            new_position = np.clip(
                new_position, LOWER, UPPER
            )

            new_wolves.append(new_position)

        wolves = np.array(new_wolves)

    # Return best solution
    scores = np.array([fitness(w) for w in wolves])
    alpha, beta, delta = find_best_wolves(wolves, scores)

    if fitness(alpha) > min(convergence):
        # Keep the best solution found during iterations
        best_iteration = np.argmin(convergence)
        # The final population is still evaluated below.

    return alpha, convergence


# Run GWO
best_wolf, convergence = gwo()

# Get optimized path
waypoints = best_wolf.reshape(WAYPOINTS, 2)
path = np.vstack([START, waypoints, GOAL])

print("Optimized Path:")
for point in path:
    print(np.round(point, 2))

print("Final Fitness:", round(fitness(best_wolf), 3))


# GRAPH 1: Robot Path Planning
plt.figure(figsize=(8, 7))

for ox, oy, radius in OBSTACLES:
    obstacle = Circle(
        (ox, oy), radius,
        color="red", alpha=0.5
    )
    plt.gca().add_patch(obstacle)

    safe_zone = Circle(
        (ox, oy), radius + CLEARANCE,
        fill=False, linestyle="--",
        color="red"
    )
    plt.gca().add_patch(safe_zone)

plt.plot(
    path[:, 0], path[:, 1],
    "b-o", linewidth=2, label="GWO Robot Path"
)

plt.scatter(*START, color="green", s=100, label="Start")
plt.scatter(*GOAL, color="black", s=100, label="Goal")

plt.xlim(0, 10)
plt.ylim(0, 10)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Robot Path Planning using GWO")
plt.legend()
plt.grid()
plt.show()


# GRAPH 2: Convergence Graph
plt.figure(figsize=(8, 5))

plt.plot(convergence, linewidth=2)

plt.xlabel("Iterations")
plt.ylabel("Fitness Value")
plt.title("GWO Convergence Graph")
plt.grid()
plt.show()

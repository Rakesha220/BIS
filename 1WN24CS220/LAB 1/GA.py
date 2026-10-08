import random
import math


# ---------------------------------------------------
# 1. CITY COORDINATES FROM THE PAPER
# ---------------------------------------------------

cities = [
    (0, 0),
    (20, 10),
    (45, 20),
    (15, 25),
    (10, 50),
    (80, 20),
    (57, 69),
    (67, 34),
    (76, 72),
    (34, 65),
    (89, 65),
    (21, 85),
    (61, 110),
    (45, 100),
    (75, 97),
    (100, 100),
    (25, 110),
    (110, 45),
    (110, 110),
    (105, 15)
]

NUM_CITIES = len(cities)

# Paper uses population of 10 chromosomes
POPULATION_SIZE = 10

# Number of generations
GENERATIONS = 500

# Mutation probability
MUTATION_RATE = 0.10

# M-point crossover
# We use 2 points as a simple M-point example
M_POINTS = 2


# ---------------------------------------------------
# 2. EUCLIDEAN DISTANCE
# ---------------------------------------------------

def distance(city1, city2):
    x1, y1 = cities[city1]
    x2, y2 = cities[city2]

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ---------------------------------------------------
# 3. CALCULATE TOTAL DISTANCE / FITNESS
# ---------------------------------------------------

def fitness(route):

    total_distance = 0

    for i in range(len(route) - 1):
        total_distance += distance(route[i], route[i + 1])

    # Return to starting city
    total_distance += distance(route[-1], route[0])

    return total_distance


# ---------------------------------------------------
# 4. INITIAL POPULATION
# ---------------------------------------------------

def create_population():

    population = []

    for _ in range(POPULATION_SIZE):

        # City 0 = starting city
        remaining_cities = list(range(1, NUM_CITIES))

        random.shuffle(remaining_cities)

        route = [0] + remaining_cities

        population.append(route)

    return population


# ---------------------------------------------------
# 5. ROULETTE WHEEL SELECTION
# ---------------------------------------------------

def roulette_selection(population):

    # Smaller distance = better solution
    # Therefore use inverse distance as selection weight

    weights = []

    for route in population:

        route_distance = fitness(route)

        weight = 1 / (route_distance + 0.000001)

        weights.append(weight)

    selected = random.choices(
        population,
        weights=weights,
        k=2
    )

    return selected[0], selected[1]


# ---------------------------------------------------
# 6. M-POINT CROSSOVER
# ---------------------------------------------------

def crossover(parent1, parent2):

    child = [None] * NUM_CITIES

    # Starting city remains fixed
    child[0] = 0

    # Choose crossover points
    points = sorted(
        random.sample(
            range(1, NUM_CITIES),
            M_POINTS
        )
    )

    # Add beginning and ending positions
    points = [0] + points + [NUM_CITIES]

    # Alternately copy sections from parents
    use_parent1 = True

    for i in range(len(points) - 1):

        start = points[i]
        end = points[i + 1]

        if use_parent1:
            source = parent1
        else:
            source = parent2

        for j in range(start, end):

            if j != 0:
                child[j] = source[j]

        use_parent1 = not use_parent1

    # ------------------------------------------------
    # Repair duplicate cities
    # ------------------------------------------------

    used = set()

    for city in child:

        if city is not None:
            used.add(city)

    missing = [
        city for city in range(1, NUM_CITIES)
        if city not in used
    ]

    missing_index = 0

    for i in range(1, NUM_CITIES):

        if child[i] is None or child.count(child[i]) > 1:

            # Find duplicate
            current_city = child[i]

            if current_city in used:
                child[i] = missing[missing_index]
                missing_index += 1
            else:
                used.add(current_city)

    return child


# ---------------------------------------------------
# 7. INTERCHANGE MUTATION
# ---------------------------------------------------

def mutation(route):

    if random.random() < MUTATION_RATE:

        # Select two positions
        pos1, pos2 = random.sample(
            range(1, NUM_CITIES),
            2
        )

        # Swap them
        route[pos1], route[pos2] = \
            route[pos2], route[pos1]

    return route


# ---------------------------------------------------
# 8. GENETIC ALGORITHM
# ---------------------------------------------------

def genetic_algorithm():

    # Create initial population
    population = create_population()

    best_route = None
    best_distance = float("inf")

    for generation in range(GENERATIONS):

        # --------------------------------------------
        # Evaluate population
        # --------------------------------------------

        for route in population:

            route_distance = fitness(route)

            if route_distance < best_distance:

                best_distance = route_distance
                best_route = route.copy()

        # --------------------------------------------
        # Create new population
        # --------------------------------------------

        new_population = []

        while len(new_population) < POPULATION_SIZE:

            # Selection
            parent1, parent2 = roulette_selection(population)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutation(child)

            new_population.append(child)

        population = new_population

        # Display progress
        if generation % 50 == 0:

            print(
                "Generation:",
                generation,
                "Best Distance:",
                round(best_distance, 2)
            )

    return best_route, best_distance


# ---------------------------------------------------
# 9. RUN PROGRAM
# ---------------------------------------------------

best_route, best_distance = genetic_algorithm()


# ---------------------------------------------------
# 10. DISPLAY FINAL RESULT
# ---------------------------------------------------

print("\n-----------------------------")
print("FINAL RESULT")
print("-----------------------------")

print("Best Route:")

for city in best_route:

    print(
        city + 1,
        cities[city]
    )

print("Return to City 1")

print(
    "\nMinimum Distance:",
    round(best_distance, 2)
)
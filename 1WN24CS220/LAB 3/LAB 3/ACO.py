
import random
import matplotlib.pyplot as plt
import networkx as nx

def aco_tsp(dist, ants=10, iterations=100):
    n = len(dist)
    alpha, beta, rho, Q = 1, 2, 0.5, 100

    tau = [[1.0 if i != j else 0.0
            for j in range(n)] for i in range(n)]

    best_tour = None
    best_distance = float("inf")

    for _ in range(iterations):
        all_tours = []

        for _ in range(ants):
            start = random.randrange(n)
            tour = [start]
            unvisited = set(range(n)) - {start}

            while unvisited:
                i = tour[-1]
                cities = sorted(unvisited)

                weights = [
                    tau[i][j]**alpha * (1 / dist[i][j])**beta
                    for j in cities
                ]

                j = random.choices(cities, weights=weights, k=1)[0]
                tour.append(j)
                unvisited.remove(j)

            tour.append(start)

            length = sum(
                dist[tour[k]][tour[k + 1]]
                for k in range(n)
            )
            all_tours.append((tour, length))

            if length < best_distance:
                best_distance = length
                best_tour = tour[:]

        # Pheromone evaporation
        for i in range(n):
            for j in range(n):
                if i != j:
                    tau[i][j] *= (1 - rho)

        # Pheromone deposition
        for tour, length in all_tours:
            for k in range(n):
                i, j = tour[k], tour[k + 1]
                tau[i][j] += Q / length
                tau[j][i] += Q / length

    return best_tour, best_distance


dist = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

tour, distance = aco_tsp(dist)

print("Best Tour:", " -> ".join(str(c+1) for c in tour))
print("Shortest Distance:", distance)

# Create graph
G = nx.Graph()
for i in range(len(dist)):
    for j in range(i + 1, len(dist)):
        G.add_edge(i+1, j+1, weight=dist[i][j])

# Draw all connections
pos = nx.spring_layout(G, seed=10)
nx.draw(G, pos, with_labels=True, node_size=1200,
        font_size=12, font_weight="bold")
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=nx.get_edge_attributes(G, "weight")
)

# Highlight best tour
route_edges = [
    (tour[i]+1, tour[i+1]+1)
    for i in range(len(tour)-1)
]
nx.draw_networkx_edges(
    G, pos, edgelist=route_edges,
    edge_color="red", width=3
)

plt.title(f"ACO Best Tour (Distance = {distance})")
plt.show()

import random
import matplotlib.pyplot as plt


def read_tsp(filename) -> list[tuple[float, float]]:
    coords = []
    reading = False
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == 'NODE_COORD_SECTION':
                reading = True
                continue
            if line == 'EOF':
                break
            if reading:
                parts = line.split()
                coords.append((float(parts[1]), float(parts[2])))
    return coords


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def tour_length(tour: list[int], coords: list[tuple[float, float]]) -> float:
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += distance(coords[tour[i]], coords[tour[(i + 1) % n]])
    return total


def evaluate(population: list[list[int]], coords: list[tuple[float, float]]) -> list[float]:
    return [tour_length(tour, coords) for tour in population]


def find_best(population: list[list[int]], scores: list[float]) -> tuple[list[int], float]:
    best_idx = scores.index(min(scores))
    best_tour = population[best_idx][:]
    best_score = scores[best_idx]
    return best_tour, best_score


def roulette_selection(population: list[list[int]], scores: list[float], n_select: int) -> list[list[int]]:
    weights = [1.0 / s for s in scores]
    return random.choices(population, weights=weights, k=n_select)


# ── Tournament selection: k random individuals compete, best one wins ────────

def tournament_selection(population: list[list[int]], scores: list[float], n_select: int, k: int = 5) -> list[list[int]]:
    selected = []
    for _ in range(n_select):
        contestants = random.sample(range(len(population)), k)
        best = contestants[0]
        for i in contestants[1:]:
            if scores[i] < scores[best]:
                best = i
        winner = best
        selected.append(population[winner][:])
    return selected


# ── OX crossover: produce one child from two parents ────────────────────────
# Copies a random segment from parent1, then fills the remaining positions
# with cities from parent2 in the order they appear.

def crossover(parent1: list[int], parent2: list[int]) -> list[int]:
    n = len(parent1)
    start, end = sorted(random.sample(range(n), 2))

    child = [-1] * n
    child[start:end + 1] = parent1[start:end + 1]

    # Fill remaining slots with parent2's cities, preserving their order
    remaining = [city for city in parent2 if city not in child]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = remaining[idx]
            idx += 1

    return child


def mutate(tour: list[int]):
    # Pick two random positions and reverse the segment between them
    i, j = sorted(random.sample(range(len(tour)), 2))
    tour[i:j + 1] = reversed(tour[i:j + 1])


def make_offspring(parents: list[list[int]], crossover_prob: float, mutation_prob: float) -> list[list[int]]:
    offspring = []
    for _ in range(len(parents)):
        p1, p2 = random.sample(parents, 2)

        child = crossover(p1, p2) if random.random() < crossover_prob else p1[:]

        if random.random() < mutation_prob:
            mutate(child)

        offspring.append(child)
    return offspring


def genetic_algorithm(
    coords,
    population_size=200,
    t_max=500,
    crossover_prob=0.9,
    mutation_prob=0.1,
    selection_type='roulette',  # 'roulette' or 'tournament'
    tournament_k=5,
):
    n_cities = len(coords)
    population = [random.sample(range(n_cities), n_cities) for _ in range(population_size)]
    scores = evaluate(population, coords)
    best_tour, best_score = find_best(population, scores)

    history = [best_score]

    for _ in range(t_max):
        if selection_type == 'tournament':
            parents = tournament_selection(population, scores, population_size, k=tournament_k)
        else:
            parents = roulette_selection(population, scores, population_size)
        population = make_offspring(parents, crossover_prob, mutation_prob)
        scores = evaluate(population, coords)

        # Sukcesja
        gen_best_tour, gen_best_score = find_best(population, scores)
        if gen_best_score < best_score:
            best_score = gen_best_score
            best_tour  = gen_best_tour

        history.append(best_score)

    return best_tour, best_score, history


# ── Plot the best tour found ─────────────────────────────────────────────────

def plot_tour(tour, coords, score, filename='best_tour.png'):
    # Close the loop by returning to the start city
    route = tour + [tour[0]]
    xs = [coords[c][0] for c in route]
    ys = [coords[c][1] for c in route]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xs, ys, color='steelblue', linewidth=1.2, zorder=1)
    ax.scatter(xs[:-1], ys[:-1], color='steelblue', s=60, zorder=2)
    ax.scatter(xs[0], ys[0], color='red', s=100, zorder=3, label='Start')

    for order, city in enumerate(tour):
        ax.annotate(str(order), xy=(coords[city][0], coords[city][1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_title(f'Best tour — length: {score:.2f}')
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ── Plot convergence across all runs ────────────────────────────────────────

def plot_convergence(all_histories, filename='convergence.png'):
    plt.figure(figsize=(10, 5))

    for history in all_histories:
        plt.plot(history, alpha=0.35, linewidth=0.8, color='steelblue')

    avg = [sum(h[i] for h in all_histories) / len(all_histories)
           for i in range(len(all_histories[0]))]
    plt.plot(avg, color='black', linewidth=2, label='Average')
    plt.axhline(y=27603, color='red', linestyle='--', linewidth=1.5, label='Optimal (27603)')

    plt.xlabel('Generation')
    plt.ylabel('Tour length')
    plt.title('GA convergence on wi29.tsp')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


# ── Run 10 times and report results ─────────────────────────────────────────

def run_experiment(coords, n_runs, **kwargs):
    all_histories = []
    all_distances = 0.0
    best_tour     = None
    best_distance = float('inf')

    for run in range(n_runs):
        tour, dist, history = genetic_algorithm(coords, **kwargs)
        print(f"  Run {run + 1:2d}: {int(dist)}")
        all_distances += dist
        all_histories.append(history)
        if dist < best_distance:
            best_distance = dist
            best_tour     = tour

    avg = all_distances / n_runs
    print(f"  Average : {int(avg)}")
    print(f"  Best    : {int(best_distance)}")
    print(f"  Optimal : 27603")
    return all_histories, best_tour, best_distance


# ── Shared helper: plot average convergence curve for each config ────────────

def plot_multi_convergence(results, title):
    """results: dict of { label: list_of_histories }"""
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10, 5))

    for (label, histories), color in zip(results.items(), colors):
        avg = [sum(h[i] for h in histories) / len(histories)
               for i in range(len(histories[0]))]
        plt.plot(avg, linewidth=2, label=label, color=color)

    plt.xlabel('Generation')
    plt.ylabel('Tour length')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ── Test 1: population size ──────────────────────────────────────────────────

def test_population_sizes(coords, n_runs=5):
    sizes = [50, 100, 200, 400]
    results = {}
    for size in sizes:
        print(f"\n── Population size: {size} ──")
        histories, _, _ = run_experiment(coords, n_runs, population_size=size,
                                         selection_type='roulette')
        results[f'size={size}'] = histories
    plot_multi_convergence(results, title='Effect of population size (roulette)')


# ── Test 2: crossover probability ───────────────────────────────────────────

def test_crossover_prob(coords, n_runs=5):
    probs = [0.5, 0.7, 0.9, 1.0]
    results = {}
    for p in probs:
        print(f"\n── Crossover prob: {p} ──")
        histories, _, _ = run_experiment(coords, n_runs, crossover_prob=p)
        results[f'pc={p}'] = histories
    plot_multi_convergence(results, title='Effect of crossover probability')


# ── Test 3: mutation probability ─────────────────────────────────────────────

def test_mutation_prob(coords, n_runs=20):
    probs = [0, 0.3, 0.6, 1.0]
    r_results = {}
    for p in probs:
        print(f"\n── Mutation prob: {p} ──")
        histories, _, _ = run_experiment(coords, n_runs, mutation_prob=p, population_size=200, t_max=500)
        r_results[f'pm={p}'] = histories
    plot_multi_convergence(r_results, title='Effect of mutation probability')

    t_results = {}
    for p in probs:
        print(f"\n── Mutation prob: {p} ──")
        histories, _, _ = run_experiment(coords, n_runs, mutation_prob=p, selection_type='tournament', population_size=200, t_max=200)
        t_results[f'pm={p}'] = histories
    plot_multi_convergence(t_results, title='Effect of mutation probability')


# ── Test 4: roulette vs tournament ───────────────────────────────────────────

def test_selection_methods(coords, n_runs=10):
    results = {}
    for sel_type in ('roulette', 'tournament'):
        print(f"\n── {sel_type.upper()} ──")
        histories, best_tour, best_dist = run_experiment(
            coords, n_runs, selection_type=sel_type, population_size=200, t_max=500)
        results[sel_type] = histories
        plot_tour(best_tour, coords, best_dist,
                  filename=f'best_tour_{sel_type}.png')
    plot_multi_convergence(results, title='Roulette vs Tournament selection')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    coords = read_tsp('wi29.tsp')

    print("Select a test to run:")
    print("  1 — Population size comparison (roulette)")
    print("  2 — Crossover probability comparison")
    print("  3 — Mutation probability comparison")
    print("  4 — Roulette vs Tournament selection")

    choice = input("\nEnter 1 / 2 / 3 / 4: ").strip()

    if   choice == '1': test_population_sizes(coords)
    elif choice == '2': test_crossover_prob(coords)
    elif choice == '3': test_mutation_prob(coords)
    elif choice == '4': test_selection_methods(coords)
    else: print("Invalid choice.")
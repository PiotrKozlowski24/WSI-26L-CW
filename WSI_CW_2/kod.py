"""
WSI CW2 - Algorytm genetyczny w problemie komiwojażera
Autor: Piotr Kozłowski
"""

import random
import time
import matplotlib.pyplot as plt
import statistics

def read_tsp(filename) -> list[tuple[float, float]]:
    coords, reading = [], False
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == 'NODE_COORD_SECTION':
                reading = True
            elif line == 'EOF':
                break
            elif reading:
                _, x, y = line.split()
                coords.append((float(x), float(y)))
    return coords


def distance(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def tour_length(tour, coords) -> float:
    n = len(tour)
    return sum(distance(coords[tour[i]], coords[tour[(i + 1) % n]]) for i in range(n))


def roulette_selection(population, scores, n):
    weights = [1.0 / s for s in scores]
    return random.choices(population, weights=weights, k=n)


def tournament_selection(population, scores, n, k=10):
    selected = []
    for _ in range(n):
        contestants = random.sample(range(len(population)), k)
        winner = min(contestants, key=lambda i: scores[i])
        selected.append(population[winner][:])
    return selected


def crossover(p1, p2) -> list[int]:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1] * n
    child[a:b + 1] = p1[a:b + 1]
    remaining = [c for c in p2 if c not in child]
    idx = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = remaining[idx]
            idx += 1
    return child


def mutate(tour):
    i, j = sorted(random.sample(range(len(tour)), 2))
    tour[i:j + 1] = tour[i:j + 1][::-1]


def genetic_algorithm(
        coords, 
        initial_population, 
        t_max=1000,
        crossover_prob=0.2, 
        mutation_prob=0.01,
        selection_type='roulette', 
        tournament_k=10
    ):
    
    population = initial_population[:]
    scores = [tour_length(t, coords) for t in population]

    best_idx = scores.index(min(scores))
    best_tour, best_score = population[best_idx][:], scores[best_idx]
    history = [best_score]

    for _ in range(t_max):
        n = len(population)
        if selection_type == 'tournament':
            parents = tournament_selection(population, scores, n, k=tournament_k)
        else:
            parents = roulette_selection(population, scores, n)

        population = []
        for _ in range(n):
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2) if random.random() < crossover_prob else p1[:]
            if random.random() < mutation_prob:
                mutate(child)
            population.append(child)

        scores = [tour_length(t, coords) for t in population]
        idx = scores.index(min(scores))
        if scores[idx] < best_score:
            best_score, best_tour = scores[idx], population[idx][:]

        history.append(best_score)

    return best_tour, best_score, history


def run_experiment(coords, initial_population, n_runs, **kwargs):
    all_histories, all_times, all_dists = [], [], []
    best_tour, best_dist = None, float('inf')

    for run in range(n_runs):
        t0 = time.time()
        tour, dist, history = genetic_algorithm(coords, initial_population, **kwargs)
        elapsed = time.time() - t0

        print(f"  Run {run + 1:2d}: {int(dist):6d}  ({elapsed:.2f}s)")

        all_histories.append(history)
        all_times.append(elapsed)
        all_dists.append(dist)

        if dist < best_dist:
            best_dist, best_tour = dist, tour

    avg_dist = sum(all_dists) / n_runs
    avg_time = sum(all_times) / n_runs
    std_dist = statistics.stdev(all_dists) if n_runs > 1 else 0

    print(f"  Average : {int(avg_dist)}")
    print(f"  Avg time: {avg_time:.2f}s")
    print(f"  Best    : {int(best_dist)}")
    print(f"  Std dist: {std_dist:.2f}")
    print(f"  Optimal : 27603")

    return all_histories, best_tour, best_dist

# RYSOWANIE WYKRESÓW ================================================

OPTIMAL = 27603


def plot_convergence(all_histories, title='Zbieżność', filename=None):
    plt.figure(figsize=(10, 5))
    for h in all_histories:
        plt.plot(h, alpha=0.3, linewidth=0.8, color='steelblue')
    avg = [sum(h[i] for h in all_histories) / len(all_histories)
           for i in range(len(all_histories[0]))]
    plt.plot(avg, color='black', linewidth=2, label='Średnia')
    plt.axhline(OPTIMAL, color='red', linestyle='--', linewidth=1.5, label=f'Optymalna ({OPTIMAL})')
    plt.xlabel('Generacja')
    plt.ylabel('Długość trasy')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()


def plot_multi_convergence(results: dict, title: str):
    """results: {label: list_of_histories}"""
    colors = plt.cm.tab10.colors
    plt.figure(figsize=(10, 5))
    for (label, histories), color in zip(results.items(), colors):
        avg = [sum(h[i] for h in histories) / len(histories)
               for i in range(len(histories[0]))]
        plt.plot(avg, linewidth=2, label=label, color=color)
    plt.axhline(OPTIMAL, color='red', linestyle='--', linewidth=1.5, label=f'Optymalna ({OPTIMAL})')
    plt.xlabel('Generacja')
    plt.ylabel('Długość trasy')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_tour(tour, coords, score, filename=None):
    route = tour + [tour[0]]
    xs = [coords[c][0] for c in route]
    ys = [coords[c][1] for c in route]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xs, ys, color='steelblue', linewidth=1.2)
    ax.scatter(xs[:-1], ys[:-1], color='steelblue', s=60, zorder=2)
    ax.scatter(xs[0], ys[0], color='red', s=100, zorder=3, label='Start')
    for order, city in enumerate(tour):
        ax.annotate(str(order), (coords[city][0], coords[city][1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_title(f'Najlepsza droga — długość: {score:.2f}')
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()


# TESTY I PORÓWNANIA PARAMETRÓW ================================================

def make_population(n_cities, size):
    return [random.sample(range(n_cities), n_cities) for _ in range(size)]


def test_population_sizes(coords, init_pop, n_runs=10):
    results = {}
    for size in [10, 29, 50, 100, 200, 400, 1000, 2000]:
        print(f"\n── Rozmiar populacji: {size} ──")
        pop = (init_pop * (size // len(init_pop) + 1))[:size]
        histories, _, _ = run_experiment(coords, pop, n_runs,
                                         selection_type='roulette')
        results[f'size={size}'] = histories
    plot_multi_convergence(results, 'Zbieżność dla różnych rozmiarów populacji')


def test_crossover_prob(coords, init_pop, n_runs=5):
    results = {}
    for p in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        print(f"\n── Crossover prob: {p} ──")
        histories, _, _ = run_experiment(coords, init_pop, n_runs,
                                         crossover_prob=p, mutation_prob=0, t_max=2000)
        results[f'pc={p}'] = histories
    plot_multi_convergence(results, 'Zbieżność dla różnych prawdopodobieństw krzyżowania')


def test_mutation_prob(coords, init_pop, n_runs=10):
    for sel in ('roulette', 'tournament'):
        results = {}
        for p in [0.01, 0.02, 0.05, 0.1, 0.2]:
            print(f"\n── {sel}, Mutation prob: {p} ──")
            histories, _, _ = run_experiment(coords, init_pop, n_runs,
                                             mutation_prob=p, crossover_prob=0.2,
                                             selection_type=sel, t_max=500)
            results[f'pm={p}'] = histories
        plot_multi_convergence(results, f'Zbieżność dla różnych prawdopodobieństw mutacji ({sel})')


def test_selection_methods(coords, init_pop, n_runs=10):
    results = {}
    for sel in ('roulette', 'tournament'):
        print(f"\n── {sel.upper()} ──")
        histories, best_tour, best_dist = run_experiment(
            coords, init_pop, n_runs,
            selection_type=sel, t_max=2000, crossover_prob=0.2, mutation_prob=0.01)
        results[sel] = histories
        plot_tour(best_tour, coords, best_dist, filename=f'best_tour_{sel}.png')
    plot_multi_convergence(results, 'Selekcja ruletkowa vs turniejowa')


def test_tournament_k(coords, init_pop, n_runs=10):
    results = {}
    for k in [2, 5, 10, 50, 100]:
        print(f"\n── Tournament k={k} ──")
        histories, _, _ = run_experiment(coords, init_pop, n_runs,
                                         selection_type='tournament',
                                         tournament_k=k, t_max=500)
        results[f'k={k}'] = histories
    plot_multi_convergence(results, 'Zbieżność dla różnych rozmiarów turnieju')


def test_normal_run(coords, init_pop, n_runs=30):
    print(f"\n── Normal run (roulette, pc=0.2, pm=0.01, t_max=2000, {n_runs} runs) ──")
    histories, best_tour, best_dist = run_experiment(
        coords, init_pop, n_runs,
        selection_type='roulette', crossover_prob=0.2, mutation_prob=0.01, t_max=50000)
    plot_tour(best_tour, coords, best_dist, filename='best_tour_normal.png')
    plot_convergence(histories, filename='convergence_normal.png')


def test_heatmap(coords, init_pop, n_runs=3):
    crossover_probs = [0.0, 0.2, 0.5, 0.7, 1.0]
    mutation_probs  = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    grid = []
    for pm in mutation_probs:
        row = []
        for pc in crossover_probs:
            print(f"\n── pc={pc}, pm={pm} ──")
            _, _, best = run_experiment(coords, init_pop, n_runs,
                                        crossover_prob=pc, mutation_prob=pm,
                                        selection_type='roulette', t_max=500)
            row.append(best)
        grid.append(row)
    grid = list(map(list, zip(*grid)))
    plt.figure(figsize=(10, 6))
    im = plt.imshow(grid, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Best distance')
    plt.xticks(range(len(mutation_probs)),  [str(p) for p in mutation_probs])
    plt.yticks(range(len(crossover_probs)), [str(p) for p in crossover_probs])
    plt.xlabel('Mutation probability')
    plt.ylabel('Crossover probability')
    plt.title('GA best distance heatmap (crossover vs mutation)')
    plt.tight_layout()
    plt.savefig('heatmap_crossover_mutation.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    coords = read_tsp('wi29.tsp')
    n_cities = len(coords)

    POPULATION_SIZE = 200
    init_pop = make_population(n_cities, POPULATION_SIZE)

    print("Select a test to run:")
    print("  1 — Population size comparison")
    print("  2 — Crossover probability comparison")
    print("  3 — Mutation probability comparison")
    print("  4 — Roulette vs Tournament selection")
    print("  5 — Tournament size (k) comparison")
    print("  6 — Normal run")
    print("  7 — Heatmap (crossover × mutation probabilities)")
    print("  8 — Population comparison plot")

    choice = input("\nEnter 1–7: ").strip()

    if   choice == '1': test_population_sizes(coords, init_pop)
    elif choice == '2': test_crossover_prob(coords, init_pop)
    elif choice == '3': test_mutation_prob(coords, init_pop)
    elif choice == '4': test_selection_methods(coords, init_pop)
    elif choice == '5': test_tournament_k(coords, init_pop)
    elif choice == '6': test_normal_run(coords, init_pop)
    elif choice == '7': test_heatmap(coords, init_pop)
    else: print("Invalid choice.")
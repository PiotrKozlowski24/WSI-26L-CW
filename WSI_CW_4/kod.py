from collections import Counter
import numpy as np
import csv
import matplotlib.pyplot as plt


def load_csv(filepath):
    rows = []
    with open(filepath, newline='') as f:
        for row in csv.reader(f):
            if row:
                rows.append(row)
    return np.array(rows)

def train_val_test_split(X, y, val_ratio=0.2, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    n_val  = int(len(X) * val_ratio)
    return (X[idx[n_test + n_val:]], X[idx[n_test:n_test + n_val]], X[idx[:n_test]],
            y[idx[n_test + n_val:]], y[idx[n_test:n_test + n_val]], y[idx[:n_test]])

def entropy(labels):
    n = len(labels)
    return -sum(c/n * np.log(c/n) for c in Counter(labels).values())

def inf_gain(data, labels, attribute):
    n = len(labels)
    weighted = sum(
        len(labels[data[:, attribute] == val]) / n *
        entropy(labels[data[:, attribute] == val])
        for val in np.unique(data[:, attribute])
    )
    return entropy(labels) - weighted


class Node:
    def __init__(self):
        self.attribute = None
        self.label = None
        self.children = {}

def id3(data, labels, attributes, max_depth, depth=0):
    node = Node()
    if len(set(labels)) == 1 or not attributes or depth == max_depth:
        node.label = Counter(labels).most_common(1)[0][0]
        return node
    best = max(attributes, key=lambda a: inf_gain(data, labels, a))
    node.attribute = best
    remaining = [a for a in attributes if a != best]
    for val in np.unique(data[:, best]):
        mask = data[:, best] == val
        node.children[val] = id3(data[mask], labels[mask], remaining, max_depth, depth + 1)
    return node

def predict_one(node, sample):
    if node.label is not None:
        return node.label
    val = sample[node.attribute]
    if val not in node.children:
        return None
    return predict_one(node.children[val], sample)

def predict(node, data):
    return np.array([predict_one(node, row) for row in data])

def count_nodes(node):
    return 1 + sum(count_nodes(c) for c in node.children.values())

def confusion_matrix(y_true, y_pred):
    classes = sorted(set(y_true))
    idx = {c: i for i, c in enumerate(classes)}
    matrix = [[0] * len(classes) for _ in range(len(classes))]
    for t, p in zip(y_true, y_pred):
        if p is not None:
            matrix[idx[t]][idx[p]] += 1
    return classes, matrix


class ID3Classifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self._tree = None

    def fit(self, X, y):
        depth = self.max_depth or len(X)
        self._tree = id3(X, y, list(range(X.shape[1])), depth)
        return self

    def predict(self, X):
        return predict(self._tree, X)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def count_nodes(self):
        return count_nodes(self._tree)


if __name__ == "__main__":
    raw = load_csv('tic-tac-toe.data')
    X, y = raw[:, :-1], raw[:, -1]

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    print(f"Trening: {len(X_train)}, Walidacja: {len(X_val)}, Test: {len(X_test)}")

    # szuka najlepszej głębokości
    best_depth, best_acc = 1, 0.0
    for d in range(1, 15):
        clf = ID3Classifier(max_depth=d)
        clf.fit(X_train, y_train)
        acc = clf.score(X_val, y_val)
        print(f"depth={d:2d}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc, best_depth = acc, d

    print(f"\nNajlepsza głębokość: {best_depth}  (val_acc={best_acc:.3f})")
    print(f"{'Depth':>5} {'Train':>8} {'Val':>8} {'Test':>8} {'Nodes':>8}")
    print("-" * 45)

    depths = list(range(1, 15))
    train_accs, val_accs, test_accs, node_counts = [], [], [], []

    for d in depths:
        clf = ID3Classifier(max_depth=d)
        clf.fit(X_train, y_train)
        tr, v, te, n = (clf.score(X_train, y_train), clf.score(X_val, y_val),
                        clf.score(X_test, y_test),   clf.count_nodes())
        train_accs.append(tr); val_accs.append(v)
        test_accs.append(te);  node_counts.append(n)
        print(f"{d:5d} {tr:8.3f} {v:8.3f} {te:8.3f} {n:8d}")

    # Test na zbiorze testowym
    final_clf = ID3Classifier(max_depth=best_depth)
    final_clf.fit(X_train, y_train)
    test_preds = final_clf.predict(X_test)
    print(f"\nTest accuracy: {final_clf.score(X_test, y_test):.3f}")
    classes, matrix = confusion_matrix(y_test, test_preds)

    # macierz pomyłek
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='Blues', vmax=max(max(r) for r in matrix) * 2)
    ax.set_title('Macierz pomyłek')
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel('Przewidziana'); ax.set_ylabel('Prawdziwa')
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i][j], ha='center', va='center', fontsize=14)
    plt.tight_layout(); plt.show()

    # wykres dokładności na wszystkich zbiorach danych
    plt.figure(figsize=(8, 4))
    plt.plot(depths, train_accs, marker='o', label='Treningowy')
    plt.plot(depths, val_accs,   marker='o', label='Walidacyjny')
    plt.plot(depths, test_accs,  marker='o', label='Testowy')
    plt.xlabel('Maksymalna głębokość'); plt.ylabel('Dokładność klasyfikacji')
    plt.title('Dokładność klasyfikatora w zależności od maksymalnej głębokości drzewa')
    plt.legend(); plt.grid(); plt.show()

    # wykres rozmiar drzewa vs głębokość
    plt.figure(figsize=(8, 4))
    plt.plot(depths, node_counts, marker='o')
    plt.xlabel('Maksymalna głębokość'); plt.ylabel('Liczba węzłów')
    plt.title('Rozmiar drzewa w zależności od maksymalnej głębokości')
    plt.grid(); plt.show()

   # test różnych stosunków podziałów danych na zbiory 'train', 'val' i 'test'
    print("\nTEST RÓŻNYCH PODZIAŁÓW DANYCH")
    split_configs = [
        (0.01, 0.01), (0.02, 0.02), (0.05, 0.05),
        (0.1, 0.1),   (0.2, 0.2),   (0.3, 0.2),
        (0.2, 0.3),   (0.1, 0.3),   (0.3, 0.3),
    ]
    seeds = [0, 7, 21, 42, 99]
    results = []

    for val_ratio, test_ratio in split_configs:
        accs, best_depths = [], []
        for seed in seeds:
            X_tr, X_v, X_te, y_tr, y_v, y_te = train_val_test_split(
                X, y, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
            bd, ba = 1, 0.0
            for d in range(1, 15):
                clf = ID3Classifier(max_depth=d)
                clf.fit(X_tr, y_tr)
                acc = clf.score(X_v, y_v)
                if acc > ba:
                    ba, bd = acc, d
            final = ID3Classifier(max_depth=bd)
            final.fit(X_tr, y_tr)
            accs.append(final.score(X_te, y_te))
            best_depths.append(bd)

        mean_acc = np.mean(accs)
        std_acc  = np.std(accs)
        mean_depth = np.mean(best_depths)
        train_size = int(len(X) * (1 - val_ratio - test_ratio))
        print(f"val={val_ratio}, test={test_ratio}, train_n={train_size}"
              f" -> acc={mean_acc:.3f}±{std_acc:.3f}, depth={mean_depth:.1f}")
        results.append((val_ratio, test_ratio, mean_acc, std_acc, mean_depth, train_size))

    split_labels = [f"v={r[0]},t={r[1]}" for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.errorbar(split_labels, [r[2] for r in results], yerr=[r[3] for r in results],
                 marker='o', capsize=4, color='steelblue')
    ax1.set_ylabel('Dokładność (test)')
    ax1.set_title('Dokładność klasyfikatora dla różnych podziałów danych')
    ax1.set_ylim(0, 1.05)
    ax1.grid()

    ax2.plot(split_labels, [r[4] for r in results], marker='o', color='darkorange')
    ax2.set_ylabel('Wybrana głębokość')
    ax2.set_title('Średnia wybrana głębokość')
    ax2.grid()

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # test odporności algorytmu na różne seedy losowości podziału danych
    print("\nTEST ODPORNOŚCI NA SEED")
    seeds = [0, 7, 13, 21, 42, 55, 77, 99, 123, 256]
    seed_val_accs, seed_test_accs, seed_best_depths = [], [], []

    print(f"{'Seed':>6} {'Depth':>6} {'ValAcc':>8} {'TestAcc':>8}")
    print("-" * 35)
    for seed in seeds:
        X_tr, X_v, X_te, y_tr, y_v, y_te = train_val_test_split(X, y, seed=seed)
        bd, ba = 1, 0.0
        for d in range(1, 15):
            clf = ID3Classifier(max_depth=d)
            clf.fit(X_tr, y_tr)
            acc = clf.score(X_v, y_v)
            if acc > ba:
                ba, bd = acc, d
        final = ID3Classifier(max_depth=bd)
        final.fit(X_tr, y_tr)
        ta = final.score(X_te, y_te)
        seed_val_accs.append(ba)
        seed_test_accs.append(ta)
        seed_best_depths.append(bd)
        print(f"{seed:6d} {bd:6d} {ba:8.3f} {ta:8.3f}")

    print(f"\nTest acc: mean={np.mean(seed_test_accs):.3f}  std={np.std(seed_test_accs):.3f}"
          f"  min={np.min(seed_test_accs):.3f}  max={np.max(seed_test_accs):.3f}")

    mean_acc = np.mean(seed_test_accs)
    std_acc  = np.std(seed_test_accs)
    xs = range(len(seeds))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # --- accuracy: val i test obok siebie ---
    width = 0.4
    bars_v = ax1.bar([x - width/2 for x in xs], seed_val_accs,  width, color='steelblue', alpha=0.7, label='Walidacja')
    bars_t = ax1.bar([x + width/2 for x in xs], seed_test_accs, width, color='darkorange', alpha=0.7, label='Test')
    ax1.axhline(mean_acc, color='red', linestyle='--', label=f'Średnia test = {mean_acc:.3f}')
    ax1.fill_between([-0.5, len(seeds) - 0.5], mean_acc - std_acc, mean_acc + std_acc,
                     color='red', alpha=0.08, label=f'±std = {std_acc:.3f}')
    for bar in bars_v:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars_t:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    ax1.set_ylabel('Dokładność'); ax1.set_ylim(0, 1.15)
    ax1.set_title('Odporność modelu na seed podziału zbioru')
    ax1.legend(); ax1.grid(axis='y')

    # --- wybrana głębokość ---
    bars_d = ax2.bar(xs, seed_best_depths, color='green', alpha=0.7)
    for bar in bars_d:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('Wybrana głębokość')
    ax2.set_title('Głębokość drzewa wybrana dla każdego seedu')
    ax2.set_yticks(range(1, max(seed_best_depths) + 2))
    ax2.grid(axis='y')

    plt.xticks(xs, seeds)
    plt.xlabel('Seed')
    plt.tight_layout()
    plt.show()
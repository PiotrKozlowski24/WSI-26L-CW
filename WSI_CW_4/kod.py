from collections import Counter
import numpy as np
import csv
import matplotlib.pyplot as plt

EXE_NUM = 100

def load_csv(filepath):
    rows = []
    with open(filepath, newline='') as f:
        for row in csv.reader(f):
            if row:
                rows.append(row)
    return np.array(rows)

def train_val_test_split(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
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


# ---------------- MAIN ----------------
if __name__ == "__main__":
    raw = load_csv('tic-tac-toe.data')
    X, y = raw[:, :-1], raw[:, -1]

    seeds = list(range(EXE_NUM))

    print(f"{'Depth':>5} {'Train':>8} {'Val':>8} {'Test':>8} {'Nodes':>8}")
    print("-" * 45)

    depths = list(range(1, X.shape[1] + 2))
    train_accs, val_accs, test_accs = [], [], []
    train_stds, val_stds, test_stds = [], [], []
    node_counts = []
    node_stds = []

    for d in depths:
        tr_list, val_list, te_list, node_list = [], [], [], []

        for seed in seeds:
            X_tr, X_v, X_te, y_tr, y_v, y_te = train_val_test_split(X, y, seed=seed)
            clf = ID3Classifier(max_depth=d)
            clf.fit(X_tr, y_tr)

            tr_list.append(clf.score(X_tr, y_tr))
            val_list.append(clf.score(X_v, y_v))
            te_list.append(clf.score(X_te, y_te))
            node_list.append(clf.count_nodes())

        train_accs.append(np.mean(tr_list))
        val_accs.append(np.mean(val_list))
        test_accs.append(np.mean(te_list))

        train_stds.append(np.std(tr_list))
        val_stds.append(np.std(val_list))
        test_stds.append(np.std(te_list))

        node_counts.append(np.mean(node_list))
        node_stds.append(np.std(node_list))

        print(f"{d:5d} m:{train_accs[-1]:8.2f} std:{train_stds[-1]:8.2f},  m:{val_accs[-1]:8.2f} std:{val_stds[-1]:8.2f}, m:{test_accs[-1]:8.2f} std:{test_stds[-1]:8.2f}, {int(node_counts[-1]):8d}")

    best_depth = depths[np.argmax(val_accs)]

    # ---------------- CONFUSION MATRICES (4 MODELS) ----------------

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, seed=42)

    def get_matrix(clf):
        preds = clf.predict(X_test)
        classes, matrix = confusion_matrix(y_test, preds)
        return np.array(matrix), classes


    # --- Train models ---
    clf_d1 = ID3Classifier(max_depth=1).fit(X_train, y_train)
    clf_d3 = ID3Classifier(max_depth=3).fit(X_train, y_train)
    clf_d9 = ID3Classifier(max_depth=9).fit(X_train, y_train)
    clf_best = ID3Classifier(max_depth=best_depth).fit(X_train, y_train)

    # --- Confusion matrices ---
    mat_d1, classes = get_matrix(clf_d1)
    mat_d3, _ = get_matrix(clf_d3)
    mat_d9, _ = get_matrix(clf_d9)
    mat_best, _ = get_matrix(clf_best)


    # ---------------- PLOTTING ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()


    def plot(ax, mat, classes, title):
        mat = np.array(mat)

        row_sums = mat.sum(axis=1, keepdims=True)
        pct = np.divide(mat, row_sums, where=row_sums != 0) * 100

        ax.imshow(mat, cmap='Blues', alpha=0.85)

        ax.set_title(title, fontsize=14)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=14)
        ax.set_yticklabels(classes, fontsize=14)

        ax.set_xlabel("Predykcje", fontsize=14)
        ax.set_ylabel("Prawdziwe", fontsize=14)

        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j, i,
                    f"{mat[i][j]}\n({pct[i][j]:.1f}%)",
                    ha='center',
                    va='center',
                    fontsize=14
                )


    plot(axes[0], mat_d1, classes, "Głębokość = 1")
    plot(axes[1], mat_d3, classes, "Głębokość = 3")
    plot(axes[2], mat_best, classes, f"Głębokość = {best_depth} (najlepsza dokł.)")
    plot(axes[3], mat_d9, classes, "Głębokość = 9")

    plt.tight_layout()
    plt.savefig("confusion_matrices_4.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------- ACCURACY PLOT ----------------
    plt.figure(figsize=(8, 4))

    plt.plot(depths, train_accs, linestyle='-', marker='o', linewidth=1.5, label='Treningowy')
    plt.fill_between(depths,
                     np.array(train_accs) - np.array(train_stds),
                     np.array(train_accs) + np.array(train_stds),
                     alpha=0.1)

    plt.plot(depths, val_accs, linestyle='--', marker='o', linewidth=2, label='Walidacyjny')
    plt.fill_between(depths,
                     np.array(val_accs) - np.array(val_stds),
                     np.array(val_accs) + np.array(val_stds),
                     alpha=0.15)

    plt.plot(depths, test_accs, linestyle='-.', marker='x', linewidth=2, label='Testowy')
    plt.fill_between(depths,
                     np.array(test_accs) - np.array(test_stds),
                     np.array(test_accs) + np.array(test_stds),
                     alpha=0.15)

    plt.xlabel('Maksymalna głębokość')
    plt.ylabel('Dokładność')
    plt.title(f'Średnia dokładność ({EXE_NUM} uruchomień)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------- TREE SIZE PLOT (FIXED) ----------------
    plt.figure(figsize=(8, 4))

    plt.plot(depths, node_counts, linestyle='-', marker='o', label='Średnia liczba węzłów')

    plt.fill_between(depths,
                     np.array(node_counts) - np.array(node_stds),
                     np.array(node_counts) + np.array(node_stds),
                     alpha=0.2)

    plt.xlabel('Maksymalna głębokość')
    plt.ylabel('Liczba węzłów')
    plt.title(f'Rozmiar drzewa ({EXE_NUM} uruchomień)')
    plt.tight_layout()
    plt.savefig("tree_size_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

    # ---------------- LEARNING CURVE EXPERIMENT ----------------
    print("\nLEARNING CURVE: ACCURACY vs TRAIN SET SIZE")

    test_ratio = 0.2
    train_sizes = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8]

    results = []

    for train_ratio in train_sizes:
        accs, depths = [], []

        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(X))

            n_test = int(len(X) * test_ratio)
            test_idx = idx[:n_test]

            remaining = idx[n_test:]

            n_train = int(len(X) * train_ratio)
            train_idx = remaining[:n_train]
            val_idx = remaining[n_train:]

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_v, y_v = X[val_idx], y[val_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            # --- choose best depth ---
            bd, ba = 1, 0.0
            for d in range(1, 15):
                clf = ID3Classifier(max_depth=d)
                clf.fit(X_tr, y_tr)
                acc = clf.score(X_v, y_v)
                if acc > ba:
                    ba, bd = acc, d

            # --- final model ---
            final = ID3Classifier(max_depth=bd)
            final.fit(X_tr, y_tr)

            accs.append(final.score(X_te, y_te))
            depths.append(bd)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)

        mean_depth = np.mean(depths)
        std_depth = np.std(depths)

        print(
            f"train_ratio={train_ratio:.2f} -> "
            f"acc={mean_acc:.3f}±{std_acc:.3f}, "
            f"depth={mean_depth:.2f}±{std_depth:.2f}"
        )

        results.append((train_ratio, mean_acc, std_acc, mean_depth, std_depth))

    train_sizes = [r[0] for r in results]
    x = np.array(train_sizes)

    mean_acc = np.array([r[1] for r in results])
    std_acc  = np.array([r[2] for r in results])

    mean_depth = np.array([r[3] for r in results])
    std_depth  = np.array([r[4] for r in results])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # -------- Accuracy --------
    ax1.plot(x, mean_acc, marker='o', linewidth=2, color='steelblue')
    ax1.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, alpha=0.25)

    ax1.set_ylabel("Dokładność")
    ax1.set_title("Dokładność klasyfikatora vs Rozmiar zbioru treningowego")
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(alpha=0.3)

    # -------- Depth --------
    ax2.plot(x, mean_depth, marker='o', linewidth=2, color='darkorange')
    ax2.fill_between(x, mean_depth - std_depth, mean_depth + std_depth, alpha=0.25,  color='darkorange')

    ax2.set_xlabel("Rozmiar zbioru treningowego")
    ax2.set_ylabel("Głębokość drzewa decyzyjnego")
    ax2.set_title("Ilość węzłów drzewa vs Rozmiar zbioru treningowego")
    ax2.grid(alpha=0.3)


    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=300, bbox_inches="tight")
    plt.show()
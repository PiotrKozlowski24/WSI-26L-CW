"""
Implementcja i badanie drzewa decyzyjngeo ID3
Autor: Piotr Kozłowski
"""

from collections import Counter
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder

EXE_NUM = 500


# Preprocessing danych

def load_csv(filepath):
    rows = []
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append(row)
    return np.array(rows)


def split_data(X, y, val_ratio=0.15, test_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))

    n_test = int(len(X) * test_ratio)
    n_val  = int(len(X) * val_ratio)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    return X[train_idx], X[val_idx], X[test_idx], y[train_idx], y[val_idx], y[test_idx]


# Entropia i przyrost informacji

def entropy(labels):
    n = len(labels)
    counts = Counter(labels)
    result = 0.0
    for count in counts.values():
        p = count / n
        result -= p * np.log(p)
    return result


def info_gain(data, labels, col):
    n = len(labels)
    total_entropy = entropy(labels)

    weighted = 0.0
    for val in np.unique(data[:, col]):
        subset = labels[data[:, col] == val]
        weighted += len(subset) / n * entropy(subset)

    return total_entropy - weighted

# Drzewo ID3

def build_tree(data, labels, attributes, max_depth, depth=0):
    majority = Counter(labels).most_common(1)[0][0]

    if len(set(labels)) == 1:
        return majority
    if len(attributes) == 0:
        return majority
    if depth == max_depth:
        return majority

    best = attributes[0]
    for a in attributes:
        if info_gain(data, labels, a) > info_gain(data, labels, best):
            best = a

    tree = {'attr': best, 'children': {}}
    remaining = [a for a in attributes if a != best]

    for val in np.unique(data[:, best]):
        mask = data[:, best] == val
        tree['children'][val] = build_tree(
            data[mask], labels[mask], remaining, max_depth, depth + 1
        )

    return tree


def predict_one(tree, sample):
    if not isinstance(tree, dict):
        return tree

    val = sample[tree['attr']]
    if val not in tree['children']:
        return None

    return predict_one(tree['children'][val], sample)


def predict(tree, data):
    predictions = []
    for row in data:
        predictions.append(predict_one(tree, row))
    return np.array(predictions)


def count_nodes(tree):
    if not isinstance(tree, dict):
        return 1
    total = 1
    for child in tree['children'].values():
        total += count_nodes(child)
    return total


def accuracy(tree, data, labels):
    preds = predict(tree, data)
    return np.mean(preds == labels)


# Macierz pomyłek

def confusion_matrix(y_true, y_pred):
    classes = sorted(set(y_true))
    idx = {c: i for i, c in enumerate(classes)}
    mat = np.zeros((len(classes), len(classes)), dtype=int)

    for t, p in zip(y_true, y_pred):
        if p is not None:
            mat[idx[t]][idx[p]] += 1

    return classes, mat

def plot_confusion(ax, mat, classes, title):
    row_sums = mat.sum(axis=1, keepdims=True)

    pct = np.divide(
        mat,
        row_sums,
        where=row_sums != 0,
        out=np.zeros_like(mat, dtype=float)
    ) * 100

    ax.imshow(mat, cmap='Blues', alpha=0.85)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predykcje")
    ax.set_ylabel("Prawdziwe")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i,
                f"{pct[i,j]:.1f}%",
                ha='center',
                va='center',
                fontsize=16
            )

# Badanie jakości drzewa ID3 i porównanie z sklearn
if __name__ == "__main__":

    raw = load_csv('tic-tac-toe.data')
    X, y = raw[:, :-1], raw[:, -1]

    seeds  = list(range(EXE_NUM))
    depths = list(range(1, X.shape[1] + 5))

    train_means = []
    train_stds  = []
    val_means   = []
    val_stds    = []
    node_means  = []
    node_stds   = []

    print(f"{'Depth':>5}  {'Train m':>8} {'Train s':>8}  {'Val m':>8} {'Val s':>8}  {'Nodes':>7}")
    print('-' * 58)

    for d in depths:
        tr_acc_list  = []
        val_acc_list = []
        node_list    = []

        for seed in seeds:
            X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y, seed=seed)

            tree = build_tree(X_tr, y_tr, list(range(X.shape[1])), max_depth=d)

            tr_acc_list.append(accuracy(tree, X_tr, y_tr))
            val_acc_list.append(accuracy(tree, X_v, y_v))
            node_list.append(count_nodes(tree))

        train_means.append(np.mean(tr_acc_list))
        train_stds.append(np.std(tr_acc_list))
        val_means.append(np.mean(val_acc_list))
        val_stds.append(np.std(val_acc_list))
        node_means.append(np.mean(node_list))
        node_stds.append(np.std(node_list))

        print(f"{d:5d}  {train_means[-1]:8.4f} {train_stds[-1]:8.4f}  "
            f"{val_means[-1]:8.4f} {val_stds[-1]:8.4f}  {int(node_means[-1]):7d}")

    best_depth = depths[np.argmax(val_means)]
    print(f"\nBest depth: {best_depth}")


    # Macierz pomyłki dla ID3

    id3_mats = []

    for seed in seeds:
        X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y, seed=seed)
        tree = build_tree(X_tr, y_tr, list(range(X.shape[1])), max_depth=best_depth)

        classes, mat = confusion_matrix(y_te, predict(tree, X_te))
        id3_mats.append(mat)

    avg_mat = np.mean(id3_mats, axis=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_confusion(ax, avg_mat, classes, f"Głębokość: {best_depth} ({EXE_NUM} uruchomień)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_best_avg.png", dpi=300)
    plt.show()


    # Wykres dokładności od głębokości

    tr = np.array(train_means)
    va = np.array(val_means)
    ts = np.array(train_stds)
    vs = np.array(val_stds)

    plt.figure(figsize=(9, 4))
    plt.plot(depths, tr, 'o-', label='Dokł. zb. tren.')
    plt.fill_between(depths, tr - ts, tr + ts, alpha=0.1)
    plt.plot(depths, va, 'o-', label='Dokł. zb. wal.')
    plt.fill_between(depths, va - vs, va + vs, alpha=0.15)
    plt.xlabel('Maksymalna głębokość')
    plt.ylabel('Średnia dokładność')
    plt.title(f'Średnia dokładność klasyfikatora ID3 w zależności od maksymalnej głębokości ({EXE_NUM} uruchomień)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png", dpi=300)
    plt.show()


    # Wykres rozmiaru drzewa od głębokości

    nm = np.array(node_means)
    ns = np.array(node_stds)

    plt.figure(figsize=(9, 4))
    plt.plot(depths, nm, 'o-', label='Średnia liczba węzłów')
    plt.fill_between(depths, nm - ns, nm + ns, alpha=0.2)
    plt.xlabel('Maksymalna głębokość')
    plt.ylabel('Liczba węzłów')
    plt.title(f'Średni rozmiar drzewa w zależności od maksymalnej głębokości ({EXE_NUM} uruchomień)')
    plt.tight_layout()
    plt.savefig("tree_size_plot.png", dpi=300)
    plt.grid(alpha=0.3)
    plt.show()



    # Porównianie ID3 i sklearn

    print("\nCOMPARISON: ID3 vs sklearn\n")

    encoder = OrdinalEncoder().fit(X)
    X_enc = encoder.transform(X)

    sk_tr_means, sk_tr_stds = [], []
    sk_val_means, sk_val_stds = [], []
    sk_node_means, sk_node_stds = [], []

    for d in depths:   # <-- SAME depths as ID3
        sk_tr_list, sk_val_list, sk_node_list = [], [], []

        for seed in seeds:
            X_tr, X_v, X_te, y_tr, y_v, y_te = split_data(X, y, seed=seed)
            X_tr_e, X_v_e, X_te_e, _, _, _ = split_data(X_enc, y, seed=seed)

            sk = DecisionTreeClassifier(
                criterion='entropy',
                max_depth=d,
                random_state=seed
            )
            sk.fit(X_tr_e, y_tr)

            sk_tr_list.append(sk.score(X_tr_e, y_tr))
            sk_val_list.append(sk.score(X_v_e, y_v))
            sk_node_list.append(sk.tree_.node_count)

        sk_tr_means.append(np.mean(sk_tr_list))
        sk_tr_stds.append(np.std(sk_tr_list))

        sk_val_means.append(np.mean(sk_val_list))
        sk_val_stds.append(np.std(sk_val_list))

        sk_node_means.append(np.mean(sk_node_list))
        sk_node_stds.append(np.std(sk_node_list))

    best_sk_depth = depths[np.argmax(sk_val_means)]

    print(f"{'Depth':>5} | "
        f"{'ID3 Train':>18} | {'SK Train':>18} | "
        f"{'ID3 Val':>18} | {'SK Val':>18} | "
        f"{'ID3 Nodes':>12} | {'SK Nodes':>12}")
    print("-" * 120)

    for i, d in enumerate(depths):
        print(
            f"{d:5d} | "
            f"{train_means[i]:.4f}±{train_stds[i]:.4f} | "
            f"{sk_tr_means[i]:.4f}±{sk_tr_stds[i]:.4f} | "
            f"{val_means[i]:.4f}±{val_stds[i]:.4f} | "
            f"{sk_val_means[i]:.4f}±{sk_val_stds[i]:.4f} | "
            f"{int(node_means[i]):12d} | "
            f"{int(sk_node_means[i]):12d}"
        )


    # Wykres z porównaniem dokładności obu drzew

    plt.figure(figsize=(10, 5))

    # ID3
    plt.plot(depths, train_means, '-', color='tab:blue', label='ID3 train', marker='o')
    plt.plot(depths, val_means, '--', color='tab:blue', label='ID3 val', marker='o')

    # sklearn
    plt.plot(depths, sk_tr_means, '-', color='tab:orange', label='sklearn train', marker='x')
    plt.plot(depths, sk_val_means, '--', color='tab:orange', label='sklearn val', marker='x')

    plt.fill_between(
        depths,
        np.array(train_means) - np.array(train_stds),
        np.array(train_means) + np.array(train_stds),
        alpha=0.1,
        color='tab:blue'
    )

    plt.fill_between(
        depths,
        np.array(val_means) - np.array(val_stds),
        np.array(val_means) + np.array(val_stds),
        alpha=0.15,
        color='tab:blue'
    )

    plt.fill_between(
        depths,
        np.array(sk_tr_means) - np.array(sk_tr_stds),
        np.array(sk_tr_means) + np.array(sk_tr_stds),
        alpha=0.1,
        color='tab:orange'
    )

    plt.fill_between(
        depths,
        np.array(sk_val_means) - np.array(sk_val_stds),
        np.array(sk_val_means) + np.array(sk_val_stds),
        alpha=0.15,
        color='tab:orange'
    )

    plt.xlabel('Maksymalna głębokość')
    plt.ylabel('Średnia dokładność klasyfikacji')
    plt.title(f'Średnia dokładność klasyfikatorów ID3 i sklearn w zależności od maksymalnej głębokości ({EXE_NUM} uruchomień)')
    plt.legend(["ID3 - zb. tren.", "ID3 - zb. wal.", "sklearn - zb. tren.", "sklearn - zb. wal."])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("cmp_accuracy.png", dpi=300)
    plt.show()


    # Wykres z porównaniem rozmiarów obu drzew

    plt.figure(figsize=(9, 5))

    id3_nodes = np.array(node_means)
    id3_nodes_std = np.array(node_stds)

    sk_nodes = np.array(sk_node_means)
    sk_nodes_std = np.array(sk_node_stds)

    # ID3
    plt.plot(depths, id3_nodes, 'o-', label='ID3')
    plt.fill_between(
        depths,
        id3_nodes - id3_nodes_std,
        id3_nodes + id3_nodes_std,
        alpha=0.2
    )

    # sklearn
    plt.plot(depths, sk_nodes, 'o-', label='sklearn')
    plt.fill_between(
        depths,
        sk_nodes - sk_nodes_std,
        sk_nodes + sk_nodes_std,
        alpha=0.2
    )

    plt.xlabel('Maksymalna głębokość')
    plt.ylabel('Średnia liczba węzłów')
    plt.title(f'Średni rozmiar drzew ID3 i sklearn w zależności od maksymalnej głębokości ({EXE_NUM} uruchomień)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("cmp_size.png", dpi=300)
    plt.show()

    print("\nBEST MODELS")
    print(f"ID3 best depth     : {best_depth}")
    print(f"sklearn best depth : {best_sk_depth}")
    print("=" * 60)


    # Macierze pomyłek dla obu modeli

    id3_mats = []
    sk_mats  = []

    for seed in seeds:
        X_tr,   X_v,   X_te,   y_tr, y_v, y_te = split_data(X,     y, seed=seed)
        X_tr_e, X_v_e, X_te_e, _,    _,   _    = split_data(X_enc, y, seed=seed)

        tree = build_tree(X_tr, y_tr, list(range(X.shape[1])), max_depth=best_depth)
        _, mat_id3 = confusion_matrix(y_te, predict(tree, X_te))
        id3_mats.append(mat_id3)

        sk = DecisionTreeClassifier(criterion='entropy', max_depth=best_sk_depth)
        sk.fit(X_tr_e, y_tr)
        _, mat_sk = confusion_matrix(y_te, sk.predict(X_te_e))
        sk_mats.append(mat_sk)

    avg_id3 = np.mean(id3_mats, axis=0)
    avg_sk  = np.mean(sk_mats,  axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    plot_confusion(axes[0], avg_id3, classes, f"ID3 (głębokość: {best_depth})")
    plot_confusion(axes[1], avg_sk,  classes, f"sklearn (głębokość: {best_sk_depth})")
    plt.tight_layout()
    plt.savefig("cmp_confusion.png", dpi=300)
    plt.show()
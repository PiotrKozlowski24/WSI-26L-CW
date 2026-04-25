from collections import Counter
import numpy as np
import csv    
import matplotlib.pyplot as plt
# ========================
# KODOWANIE TEKSTU NA LICZBY
# ========================

def fit_encoders(data):
    """Dla każdej kolumny tworzy słownik {wartość_tekstowa: liczba}"""
    encoders = []
    for col in range(data.shape[1]):
        unique_vals = sorted(set(data[:, col]))
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        encoders.append(mapping)
    return encoders

def apply_encoders(data, encoders):
    """Zamienia tekstowe wartości na liczby wg słowników"""
    encoded = np.empty(data.shape, dtype=int)
    for col, mapping in enumerate(encoders):
        for row in range(data.shape[0]):
            encoded[row, col] = mapping[data[row, col]]
    return encoded

# ========================
# WCZYTANIE DANYCH
# ========================

def load_csv(filepath):
    """Wczytuje CSV i zwraca numpy array stringów"""
    rows = []
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append(row)
    return np.array(rows)

# ========================
# PODZIAŁ DANYCH
# ========================

def train_val_test_split(X, y, val_ratio=0.3, test_ratio=0.3, seed=42):
    """Dzieli dane na zbiór treningowy, walidacyjny i testowy"""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    n = len(X)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)

    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    return X[train_idx], X[val_idx], X[test_idx], \
           y[train_idx], y[val_idx], y[test_idx]

# ========================
# ENTROPIA I INFORMATION GAIN
# ========================

def entropy(labels):
    n = len(labels)
    counts = Counter(labels)
    result = 0.0
    for c in counts.values():
        if c > 0:
            f = c / n
            result -= f * np.log(f)
    return result

def inf(data, labels, attribute):
    n = len(labels)
    result = 0.0
    for val in np.unique(data[:, attribute]):
        mask = data[:, attribute] == val
        U_j = labels[mask]
        weight = len(U_j) / n
        result += weight * entropy(U_j)
    return result

def inf_gain(data, labels, attribute):
    return entropy(labels) - inf(data, labels, attribute)

# ========================
# DRZEWO DECYZYJNE ID3
# ========================

class Node:
    def __init__(self):
        self.attribute = None
        self.label = None
        self.children = {}

def id3(data, labels, attributes, max_depth, depth=0):
    node = Node()

    # warunek 1: wszystkie etykiety takie same
    if len(set(labels)) == 1:
        node.label = labels[0]
        return node

    # warunek 2: brak atrybutów lub osiągnięto max głębokość
    if len(attributes) == 0 or depth == max_depth:
        node.label = Counter(labels).most_common(1)[0][0]
        return node

    # wybierz atrybut z największym zyskiem informacyjnym
    best = max(attributes, key=lambda a: inf_gain(data, labels, a))
    node.attribute = best
    remaining = [a for a in attributes if a != best]

    for val in np.unique(data[:, best]):
        mask = data[:, best] == val
        if mask.sum() == 0:
            child = Node()
            child.label = Counter(labels).most_common(1)[0][0]
        else:
            child = id3(data[mask], labels[mask], remaining, max_depth, depth + 1)
        node.children[val] = child

    return node

# ========================
# PREDYKCJA
# ========================

def predict_one(node, sample):
    if node.label is not None:
        return node.label
    val = sample[node.attribute]
    if val not in node.children:
        return None
    return predict_one(node.children[val], sample)

def predict(node, data):
    return np.array([predict_one(node, row) for row in data])

# ========================
# METRYKI
# ========================

def accuracy(y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)

def confusion_matrix(y_true, y_pred):
    classes = sorted(set(y_true))
    class_index = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if p is not None:
            matrix[class_index[t]][class_index[p]] += 1
    return classes, matrix

def print_confusion_matrix(classes, matrix):
    print("Macierz pomyłek (wiersze=prawdziwe, kolumny=przewidziane):")
    header = f"{'':>12}" + "".join(f"{str(c):>12}" for c in classes)
    print(header)
    for i, row in enumerate(matrix):
        line = f"{str(classes[i]):>12}" + "".join(f"{v:>12}" for v in row)
        print(line)

# ========================
# GŁÓWNY PROGRAM
# ========================

if __name__ == "__main__":
    # wczytaj dane
    raw = load_csv('tic-tac-toe.data')
    data_str = raw[:, :-1]   # atrybuty jako stringi
    labels_str = raw[:, -1]  # klasy jako stringi

    # zakoduj atrybuty i klasy na liczby
    encoders = fit_encoders(data_str)
    X = apply_encoders(data_str, encoders)

    label_encoder = fit_encoders(labels_str.reshape(-1, 1))
    y_encoded = apply_encoders(labels_str.reshape(-1, 1), label_encoder)
    y = y_encoded[:, 0]

    # podział danych
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_ratio=0.2, test_ratio=0.2, seed=42
    )
    print(f"Trening: {len(X_train)}, Walidacja: {len(X_val)}, Test: {len(X_test)}")

    # szukaj najlepszej głębokości
    attributes = list(range(X_train.shape[1]))
    best_depth, best_acc = 1, 0.0

    for max_depth in range(1, 15):
        tree = id3(X_train, y_train, attributes, max_depth)
        preds = predict(tree, X_val)
        acc = accuracy(y_val, preds)
        print(f"depth={max_depth:2d}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc, best_depth = acc, max_depth

    print(f"\nNajlepsza głębokość: {best_depth}  (val_acc={best_acc:.3f})")

    # finalny test
    final_tree = id3(X_train, y_train, attributes, best_depth)
    test_preds = predict(final_tree, X_test)

    print(f"\nTest accuracy: {accuracy(y_test, test_preds):.3f}")
    classes, matrix = confusion_matrix(y_test, test_preds)
    print_confusion_matrix(classes, matrix)

    # --- wykres 1: accuracy vs depth ---
    depths = list(range(1, 15))
    val_accs = [accuracy(y_val, predict(id3(X_train, y_train, list(range(9)), d), X_val)) for d in depths]

    plt.figure(figsize=(8, 4))
    plt.plot(depths, val_accs, marker='o', label='Walidacja')
    plt.xlabel('max_depth')
    plt.ylabel('Dokładność')
    plt.title('Dokładność vs głębokość drzewa')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()

    # --- wykres 2: macierz pomyłek jako heatmapa ---
    classes, matrix = confusion_matrix(y_test, test_preds)
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel('Przewidziana klasa')
    ax.set_ylabel('Prawdziwa klasa')
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i][j], ha='center', va='center', color='black')
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()
import pandas as pd
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np


def experiment_distances():
    dims = [2**i for i in range(11)]
    n_points = 100

    mean_l2, std_l2 = [], []
    mean_l1, std_l1 = [], []

    for d in dims:
        X = np.random.rand(n_points, d)
        diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]

        dists_l2 = np.sum(diffs**2, axis=2)
        dists_l1 = np.sum(np.abs(diffs), axis=2)

        triu_idx = np.triu_indices(n_points, k=1)
        l2_vals = dists_l2[triu_idx]
        l1_vals = dists_l1[triu_idx]

        mean_l2.append(np.mean(l2_vals))
        std_l2.append(np.std(l2_vals))

        mean_l1.append(np.mean(l1_vals))
        std_l1.append(np.std(l1_vals))

        # In kết quả cho từng dimension
        print(f"d={d}: L2^2 mean={mean_l2[-1]:.4f}, std={std_l2[-1]:.4f} | "
              f"L1 mean={mean_l1[-1]:.4f}, std={std_l1[-1]:.4f}")

    # Vẽ kết quả
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(dims, mean_l2, 'o-', label='Mean (ℓ2 squared)')
    plt.plot(dims, std_l2, 's-', label='Std (ℓ2 squared)')
    plt.xscale('log', base=2)
    plt.xlabel("Dimension d (log scale)")
    plt.ylabel("Value")
    plt.title("ℓ2 squared distance")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(dims, mean_l1, 'o-', label='Mean (ℓ1)')
    plt.plot(dims, std_l1, 's-', label='Std (ℓ1)')
    plt.xscale('log', base=2)
    plt.xlabel("Dimension d (log scale)")
    plt.ylabel("Value")
    plt.title("ℓ1 distance")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    experiment_distances()
    
def load_data():
    real_lines = pd.read_csv("clean_real.txt", header=None, names=["headlines"])
    fake_lines = pd.read_csv("clean_fake.txt", header=None, names=["headlines"])

    fake_lines["label"] = 0
    real_lines["label"] = 1

    data = pd.concat([real_lines, fake_lines], axis=0, ignore_index=True)

    X = data["headlines"]
    y = data["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.7, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_converted = vectorizer.fit_transform(X_train)
    X_val_converted = vectorizer.transform(X_val)
    X_test_converted = vectorizer.transform(X_test)

    return X_train_converted, X_val_converted, X_test_converted, y_train, y_val, y_test, vectorizer


def select_model(X_train, y_train, X_val, y_val, X_test, y_test):
    max_depth = [2, 5, 10, 20, None]
    criteria = ["entropy", "log_loss", "gini"]

    results = []

    for criterion in criteria:
        val_accuracies = []

        for depth in max_depth:
            clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
            clf.fit(X_train, y_train)
            y_val_pred = clf.predict(X_val)

            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_accuracies.append(val_accuracy)
            results.append((criterion, depth, val_accuracy))

            print(f"criterion: {criterion}, depth: {depth}, val_accuracy: {val_accuracy}")


        styles = {
            "entropy": dict(marker="o", linestyle="-", color="red"),
            "log_loss": dict(marker="s", linestyle="--", color="blue"),
            "gini": dict(marker="^", linestyle="-.", color="green"),
        }
        depths_for_plot = [d if d is not None else "None" for d in max_depth]
        plt.plot(depths_for_plot, val_accuracies, label=criterion, **styles[criterion])

    plt.xlabel("max_depth")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. max_depth")
    plt.legend(title="criterion")
    plt.grid(True, linestyle=":")
    plt.show()

    #Find best parameters
    best_param = max(results, key=lambda x: x[2])
    best_criterion, best_depth, best_acc = best_param
    print(f"\nBest model: criterion={best_criterion}, max_depth={best_depth}, val_acc={best_acc}")

    #Fit model with the best parameters
    best_clf = DecisionTreeClassifier(criterion=best_criterion, max_depth=best_depth, random_state=42)
    best_clf.fit(X_train, y_train)
    y_test_pred = best_clf.predict(X_test)

    #Accuracy of best fit on test set and confusion matrix
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy}")
    disp = ConfusionMatrixDisplay.from_estimator(
        best_clf,
        X_test,
        y_test,
        display_labels=["fake news", "real news"],
    )
    disp.plot()

    return best_clf, best_criterion, best_depth, test_accuracy


def visualize_tree(best_clf, vectorizer):
    plt.figure(figsize=(15, 8))
    plot_tree(
        best_clf,
        max_depth=2,
        filled=True,
        rounded=True,
        class_names=["fake news", "real news"],
        feature_names=vectorizer.get_feature_names_out(),
    )
    plt.show()


def entropy(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

def compute_information_gain(X, y, keyword, vectorizer):
    if keyword not in vectorizer.vocabulary_:
        raise ValueError(f"Keyword '{keyword}' not in vocabulary")

    idx = vectorizer.vocabulary_[keyword]
    feature_column = X[:, idx].toarray().ravel() > 0

    y_present = y[feature_column]
    y_absent = y[~feature_column]

    H_y = entropy(y)
    H_present = entropy(y_present) if len(y_present) > 0 else 0
    H_absent = entropy(y_absent) if len(y_absent) > 0 else 0

    H_cond = (len(y_present)/len(y)) * H_present + (len(y_absent)/len(y)) * H_absent

    IG = H_y - H_cond
    return IG




if __name__ == "__main__":
    #a)
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = load_data()

    #b)
    best_clf, best_criterion, best_depth, test_acc = select_model(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    print("Output of select_model function:", best_clf, best_criterion, best_depth, test_acc)

    #c)
    visualize_tree(best_clf, vectorizer)

    #d)
    root_feature_idx = best_clf.tree_.feature[0]
    if root_feature_idx != -2:
        root_feature_name = vectorizer.get_feature_names_out()[root_feature_idx]
        root_feature_ig = compute_information_gain(X_train, y_train, root_feature_name, vectorizer)
        print(f"Topmost split is on: '{root_feature_name}' | IG = {root_feature_ig}")

    keywords = ["trump", "hillary", "the", "s"]
    for keyword in keywords:
        try:
            ig = compute_information_gain(X_train, y_train, keyword, vectorizer)
            print(f"Keyword = {keyword:20s} | IG = {ig}")
        except ValueError as e:
            print(e)
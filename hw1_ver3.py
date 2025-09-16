import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from math import log2


def load_data(fake_file="clean_fake.txt", real_file="clean_real.txt"):
    with open(fake_file, "r", encoding="utf-8") as f:
        fake_lines = f.readlines()
    with open(real_file, "r", encoding="utf-8") as f:
        real_lines = f.readlines()

    X = fake_lines + real_lines
    y = [0] * len(fake_lines) + [1] * len(real_lines)  # 0=fake, 1=real

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_vec, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer


# Ý a: in thông số dataset và kết quả cây quyết định cơ bản
def dataset_and_baseline(X_train, X_val, X_test, y_train, y_val, y_test, vectorizer):
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    print(f"Total examples: {total}")
    print(f"Training set: {X_train.shape[0]} examples ({100*X_train.shape[0]/total:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} examples ({100*X_val.shape[0]/total:.1f}%)")
    print(f"Test set: {X_test.shape[0]} examples ({100*X_test.shape[0]/total:.1f}%)")
    print(f"Number of features: {X_train.shape[1]}")

    def count_classes(y):
        fake = sum(1 for i in y if i == 0)
        real = sum(1 for i in y if i == 1)
        return fake, real

    tr_fake, tr_real = count_classes(y_train)
    val_fake, val_real = count_classes(y_val)
    te_fake, te_real = count_classes(y_test)

    print(f"\nTraining set - Fake: {tr_fake}, Real: {tr_real}")
    print(f"Validation set - Fake: {val_fake}, Real: {val_real}")
    print(f"Test set - Fake: {te_fake}, Real: {te_real}")

    # Baseline Decision Tree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, clf.predict(X_train))
    acc_val = accuracy_score(y_val, clf.predict(X_val))
    acc_test = accuracy_score(y_test, clf.predict(X_test))

    print("\nBasic Decision Tree Results:")
    print(f"Training accuracy: {acc_train:.3f}")
    print(f"Validation accuracy: {acc_val:.3f}")
    print(f"Test accuracy: {acc_test:.3f}")


# Ý b: chọn mô hình
def select_model(X_train, y_train, X_val, y_val, X_test, y_test):
    criteria = ["gini", "entropy", "log_loss"]
    max_depths = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    results = {}

    for crit in criteria:
        accs = []
        for d in max_depths:
            clf = DecisionTreeClassifier(criterion=crit, max_depth=d, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            accs.append(acc)
            print(f"Criterion={crit}, max_depth={d}, val_acc={acc:.4f}")
        results[crit] = accs

    plt.figure(figsize=(8, 6))
    for crit in criteria:
        plt.plot(max_depths, results[crit], marker="o", label=crit)
    plt.xlabel("max_depth")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs max_depth")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Chọn mô hình tốt nhất
    best_crit, best_depth, best_acc = None, None, 0
    for crit in criteria:
        for d, acc in zip(max_depths, results[crit]):
            if acc > best_acc:
                best_acc = acc
                best_crit = crit
                best_depth = d

    clf_best = DecisionTreeClassifier(criterion=best_crit, max_depth=best_depth, random_state=42)
    clf_best.fit(np.vstack([X_train.toarray(), X_val.toarray()]), y_train + y_val)
    y_test_pred = clf_best.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Best model: criterion={best_crit}, max_depth={best_depth}, val_acc={best_acc:.4f}, test_acc={test_acc:.4f}")
    return clf_best, best_crit, best_depth


# Ý c: vẽ 2 tầng đầu của cây
def visualize_tree(clf, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    plt.figure(figsize=(12, 6))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["Fake", "Real"],
        filled=True,
        max_depth=2,
    )
    plt.title("First Two Layers of Decision Tree")
    plt.show()


# Ý d: hàm IG và phân tích chi tiết
def entropy(labels):
    total = len(labels)
    if total == 0:
        return 0
    counts = np.bincount(labels)
    probs = counts / total
    return -sum(p * log2(p) for p in probs if p > 0)


def compute_information_gain(X_train, y_train, vectorizer, keyword):
    if keyword not in vectorizer.vocabulary_:
        return None
    
    idx = vectorizer.vocabulary_[keyword]
    has_kw = X_train[:, idx].toarray().ravel() > 0
    left = [y_train[i] for i in range(len(y_train)) if has_kw[i]]
    right = [y_train[i] for i in range(len(y_train)) if not has_kw[i]]

    H_parent = entropy(y_train)
    H_left = entropy(left)
    H_right = entropy(right)

    IG = H_parent - (len(left) / len(y_train)) * H_left - (len(right) / len(y_train)) * H_right
    return IG


def analyze_information_gain(X_train, y_train, vectorizer, clf_best):
    print("\nInformation Gain Analysis:")
    print("=" * 50)
    
    # Tính information gain cho tất cả các features
    feature_names = vectorizer.get_feature_names_out()
    feature_importances = clf_best.feature_importances_
    
    ig_scores = []
    for i, feature in enumerate(feature_names):
        ig = compute_information_gain(X_train, y_train, vectorizer, feature)
        if ig is not None:
            ig_scores.append((feature, ig, feature_importances[i]))
    
    # Sắp xếp theo information gain giảm dần
    ig_scores.sort(key=lambda x: x[1], reverse=True)
    
    # In top 5 features
    for i in range(min(5, len(ig_scores))):
        feature, ig, importance = ig_scores[i]
        print(f"Top feature {i+1}: '{feature}'")
        print(f"  Information Gain: {ig:.6f}")
        print(f"  Feature Importance: {importance:.6f}")
        print()
    
    # Phân tích các keywords cụ thể
    specific_keywords = ['trump', 'news', 'media', 'fake', 'real']
    
    print("Information Gain for specific keywords:")
    print("=" * 50)
    
    for keyword in specific_keywords:
        ig = compute_information_gain(X_train, y_train, vectorizer, keyword)
        if keyword in vectorizer.vocabulary_:
            idx = vectorizer.vocabulary_[keyword]
            importance = feature_importances[idx]
        else:
            importance = 0.0
            
        if ig is not None:
            print(f"Keyword: '{keyword}'")
            print(f"  Information Gain: {ig:.6f}")
            print(f"  Feature Importance: {importance:.6f}")
            print()
        else:
            print(f"Keyword: '{keyword}'")
            print(f"  Not found in vocabulary")
            print(f"  Information Gain: N/A")
            print(f"  Feature Importance: 0.000000")
            print()


# ========== Main ==========
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = load_data()
    dataset_and_baseline(X_train, X_val, X_test, y_train, y_val, y_test, vectorizer)  # Ý a
    clf_best, best_crit, best_depth = select_model(X_train, y_train, X_val, y_val, X_test, y_test)  # Ý b
    visualize_tree(clf_best, vectorizer)  # Ý c
    
    # Ý d: Phân tích Information Gain chi tiết
    analyze_information_gain(X_train, y_train, vectorizer, clf_best)
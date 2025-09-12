import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """
    Load the data, preprocess it using a vectorizer, and split into train/validation/test sets.
    
    Returns:
        X_train, X_val, X_test: Feature matrices for training, validation, and test sets
        y_train, y_val, y_test: Labels for training, validation, and test sets
        vectorizer: The fitted vectorizer for future use
    """
    
    # Load the real news headlines
    try:
        with open('clean_real.txt', 'r', encoding='utf-8') as f:
            real_headlines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Warning: clean_real.txt not found. Using sample data for demonstration.")
        real_headlines = [
            "trump announces new policy changes",
            "president trump meets with foreign leaders",
            "trump administration releases statement",
            "trump twitter account suspended temporarily"
        ] * 500  # Multiply to simulate larger dataset
    
    # Load the fake news headlines
    try:
        with open('clean_fake.txt', 'r', encoding='utf-8') as f:
            fake_headlines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Warning: clean_fake.txt not found. Using sample data for demonstration.")
        fake_headlines = [
            "trump secretly alien lizard person revealed",
            "shocking trump scandal media won't report",
            "trump destroys deep state single handedly",
            "trump magical powers confirmed scientists"
        ] * 325  # Multiply to simulate larger dataset
    
    # Combine headlines and create labels
    all_headlines = real_headlines + fake_headlines
    labels = [1] * len(real_headlines) + [0] * len(fake_headlines)  # 1 for real, 0 for fake
    
    print(f"Loaded {len(real_headlines)} real headlines and {len(fake_headlines)} fake headlines")
    
    # Initialize vectorizer - you can experiment with different options
    # CountVectorizer counts word occurrences
    # TfidfVectorizer uses TF-IDF weighting
    vectorizer = CountVectorizer(
        max_features=5000,  # Limit vocabulary size
        stop_words='english',  # Remove common English stop words
        lowercase=True,  # Convert to lowercase
        ngram_range=(1, 2)  # Use unigrams and bigrams
    )
    
    # Alternative vectorizer option (uncomment to use):
    # vectorizer = TfidfVectorizer(
    #     max_features=5000,
    #     stop_words='english',
    #     lowercase=True,
    #     ngram_range=(1, 2)
    # )
    
    # Fit and transform the data
    X = vectorizer.fit_transform(all_headlines)
    y = np.array(labels)
    
    # First split: 70% train, 30% temp (which will be split into 15% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Second split: Split the 30% temp into 15% validation and 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set size: {X_train.shape[0]} ({X_train.shape[0]/len(all_headlines)*100:.1f}%)")
    print(f"Validation set size: {X_val.shape[0]} ({X_val.shape[0]/len(all_headlines)*100:.1f}%)")
    print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(all_headlines)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer

def train_decision_tree(X_train, y_train, **kwargs):
    """
    Train a decision tree classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        **kwargs: Additional parameters for DecisionTreeClassifier
    
    Returns:
        Trained DecisionTreeClassifier
    """
    
    # Initialize decision tree with default or custom parameters
    dt_classifier = DecisionTreeClassifier(
        random_state=42,
        **kwargs
    )
    
    # Train the classifier
    dt_classifier.fit(X_train, y_train)
    
    return dt_classifier

def evaluate_model(classifier, X, y, dataset_name="Dataset"):
    """
    Evaluate the model on a given dataset.
    
    Args:
        classifier: Trained classifier
        X: Features
        y: True labels
        dataset_name: Name of the dataset for printing
    
    Returns:
        accuracy: Accuracy score
    """
    
    predictions = classifier.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    print(f"\n{dataset_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y, predictions, target_names=['Fake', 'Real']))
    
    return accuracy

def hyperparameter_tuning(X_train, X_val, y_train, y_val):
    """
    Perform hyperparameter tuning using the validation set.
    
    Args:
        X_train, X_val: Training and validation features
        y_train, y_val: Training and validation labels
    
    Returns:
        best_params: Dictionary of best parameters
        best_accuracy: Best validation accuracy achieved
    """
    
    # Parameters to test
    param_combinations = [
        {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'max_depth': 20, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'max_depth': None, 'min_samples_split': 5, 'min_samples_leaf': 1},
        {'max_depth': None, 'min_samples_split': 10, 'min_samples_leaf': 1},
        {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 5},
        {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 10},
        {'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2},
    ]
    
    best_accuracy = 0
    best_params = None
    results = []
    
    print("\nHyperparameter Tuning Results:")
    print("-" * 50)
    
    for params in param_combinations:
        # Train model with current parameters
        dt = train_decision_tree(X_train, y_train, **params)
        
        # Evaluate on validation set
        val_pred = dt.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        results.append({**params, 'val_accuracy': val_accuracy})
        
        print(f"Params: {params}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("-" * 30)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = params
    
    print(f"\nBest Parameters: {best_params}")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    
    return best_params, best_accuracy, results

def visualize_results(results):
    """
    Visualize hyperparameter tuning results.
    
    Args:
        results: List of dictionaries containing parameter combinations and accuracies
    """
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Max depth vs accuracy
    max_depth_results = df.groupby('max_depth')['val_accuracy'].mean().reset_index()
    max_depth_results['max_depth'] = max_depth_results['max_depth'].fillna('None')
    axes[0].bar(range(len(max_depth_results)), max_depth_results['val_accuracy'])
    axes[0].set_xlabel('Max Depth')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title('Max Depth vs Accuracy')
    axes[0].set_xticks(range(len(max_depth_results)))
    axes[0].set_xticklabels(max_depth_results['max_depth'])
    
    # Plot 2: Min samples split vs accuracy
    split_results = df.groupby('min_samples_split')['val_accuracy'].mean().reset_index()
    axes[1].bar(range(len(split_results)), split_results['val_accuracy'])
    axes[1].set_xlabel('Min Samples Split')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Min Samples Split vs Accuracy')
    axes[1].set_xticks(range(len(split_results)))
    axes[1].set_xticklabels(split_results['min_samples_split'])
    
    # Plot 3: Min samples leaf vs accuracy
    leaf_results = df.groupby('min_samples_leaf')['val_accuracy'].mean().reset_index()
    axes[2].bar(range(len(leaf_results)), leaf_results['val_accuracy'])
    axes[2].set_xlabel('Min Samples Leaf')
    axes[2].set_ylabel('Validation Accuracy')
    axes[2].set_title('Min Samples Leaf vs Accuracy')
    axes[2].set_xticks(range(len(leaf_results)))
    axes[2].set_xticklabels(leaf_results['min_samples_leaf'])
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the complete pipeline.
    """
    
    print("Loading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = load_data()
    
    # Train initial model with default parameters
    print("\nTraining initial decision tree with default parameters...")
    dt_default = train_decision_tree(X_train, y_train)
    
    # Evaluate on all sets
    train_acc = evaluate_model(dt_default, X_train, y_train, "Training")
    val_acc = evaluate_model(dt_default, X_val, y_val, "Validation")
    test_acc = evaluate_model(dt_default, X_test, y_test, "Test")
    
    # Hyperparameter tuning
    print("\nStarting hyperparameter tuning...")
    best_params, best_val_acc, results = hyperparameter_tuning(X_train, X_val, y_train, y_val)
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    dt_best = train_decision_tree(X_train, y_train, **best_params)
    
    # Final evaluation
    print("\nFinal Model Performance:")
    final_train_acc = evaluate_model(dt_best, X_train, y_train, "Training (Best Model)")
    final_val_acc = evaluate_model(dt_best, X_val, y_val, "Validation (Best Model)")
    final_test_acc = evaluate_model(dt_best, X_test, y_test, "Test (Best Model)")
    
    # Visualize results
    visualize_results(results)
    
    # Feature importance analysis
    feature_names = vectorizer.get_feature_names_out()
    importance_scores = dt_best.feature_importances_
    
    # Get top 10 most important features
    top_features_idx = np.argsort(importance_scores)[-10:]
    top_features = [(feature_names[i], importance_scores[i]) for i in top_features_idx]
    top_features.reverse()  # Sort in descending order
    
    print("\nTop 10 Most Important Features:")
    for feature, importance in top_features:
        print(f"{feature}: {importance:.4f}")
    
    return dt_best, vectorizer

if __name__ == "__main__":
    # Run the main pipeline
    best_model, vectorizer = main()
    
    # Example of how to use the trained model for prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    # Test with some sample headlines
    sample_headlines = [
        "Trump announces new economic policy",
        "Scientists discover Trump has magical powers",
        "President signs new legislation today",
        "Shocking revelation media doesn't want you to know"
    ]
    
    # Transform the sample headlines using the same vectorizer
    sample_features = vectorizer.transform(sample_headlines)
    predictions = best_model.predict(sample_features)
    probabilities = best_model.predict_proba(sample_features)
    
    for i, headline in enumerate(sample_headlines):
        pred_label = "Real" if predictions[i] == 1 else "Fake"
        confidence = max(probabilities[i])
        print(f"Headline: '{headline}'")
        print(f"Prediction: {pred_label} (confidence: {confidence:.3f})")
        print("-" * 40)
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Load the data functionally
def load_data():
    from ucimlrepo import fetch_ucirepo
    iris = fetch_ucirepo(id=53)  # Fetch dataset
    X = iris.data.features
    y = iris.data.targets
    return X, y

# Train-test split
def split_data(X, y):
    return train_test_split(X, y.values.ravel(), test_size=0.2, random_state=42)

# Train the KNN model
def train_knn(X_train, y_train, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train.ravel())
    return knn

# Evaluate the model
def evaluate_model(knn, X_test, y_test):
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    return accuracy

# Perform cross-validation
def cross_validate_knn(X, y, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y.values.ravel(), cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean()}")

# Hyperparameter tuning using grid search
def grid_search_knn(X_train, y_train):
    param_grid = {'n_neighbors': range(1, 11)}  # Try k values from 1 to 10
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train.ravel())
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")
    return grid_search.best_estimator_

# Main workflow
if __name__ == "__main__":
    # Load and split data
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model with k=3
    knn = train_knn(X_train, y_train, k=3)
    
    # Evaluate the model
    evaluate_model(knn, X_test, y_test)
    
    # Cross-validation to assess generalization
    cross_validate_knn(X, y, k=3)
    
    # Hyperparameter tuning using Grid Search
    best_knn = grid_search_knn(X_train, y_train)
    
    # Evaluate the best model found by grid search
    evaluate_model(best_knn, X_test, y_test)

    # Define parameter grid for Grid Search (trying different values for n_neighbors)
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

    # Apply GridSearchCV for KNN
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train.ravel())

    # Best hyperparameters and score
    print("Best parameters found by Grid Search:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    # Re-evaluate the model with the best parameters
    best_knn = grid_search.best_estimator_

    # Predictions with optimized KNN
    y_pred_optimized = best_knn.predict(X_test)

    # Confusion Matrix, as in how many of the predictions were TP, FP, things like that
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_optimized))

    # Classification Report (precision, recall, F1-score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_optimized))

    # Visualize confusion matrix
    ConfusionMatrixDisplay.from_estimator(best_knn, X_test, y_test)
    plt.show()

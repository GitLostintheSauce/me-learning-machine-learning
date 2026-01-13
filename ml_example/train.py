import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

def train():
    # 1. Load Data
    # The Iris dataset is a classic dataset for classification.
    # It contains 150 samples of iris flowers with 4 features:
    # sepal length, sepal width, petal length, petal width.
    # The target is the species: setosa, versicolor, or virginica.
    print("Loading data...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # 2. Split Data
    # We split data into training and testing sets.
    # Training set is used to teach the model.
    # Testing set is used to evaluate how well it performs on unseen data.
    # random_state ensures reproducibility (same split every time we run).
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Initialize Model
    # We use a Decision Tree Classifier. It's simple and interpretable.
    model = DecisionTreeClassifier(random_state=42)

    # 4. Train Model (Fit)
    # The model learns the relationship between features (X) and target (y).
    print("Training model...")
    model.fit(X_train, y_train)

    # 5. Evaluate
    # We predict species for the test set and compare with actual labels.
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # 6. Save Model
    # We save the trained model to a file so we can use it later without retraining.
    print("Saving model to 'iris_model.joblib'...")
    joblib.dump(model, 'ml_example/iris_model.joblib')
    print("Done!")

if __name__ == "__main__":
    train()

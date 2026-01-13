import joblib
import pandas as pd
from sklearn.datasets import load_iris

def predict():
    # 1. Load Model
    # Load the previously trained model from the file.
    print("Loading model...")
    try:
        model = joblib.load('ml_example/iris_model.joblib')
    except FileNotFoundError:
        print("Model file not found. Please run train.py first.")
        return

    # 2. Prepare New Data
    # Let's create a sample observation (a single flower).
    # Values correspond to: sepal length, sepal width, petal length, petal width
    # This sample is likely a 'setosa' (class 0).
    new_data = [[5.1, 3.5, 1.4, 0.2]]
    
    # We need the feature names to silence warnings from pandas/sklearn alignment
    iris = load_iris()
    feature_names = iris.feature_names
    
    X_new = pd.DataFrame(new_data, columns=feature_names)

    # 3. Make Prediction
    print(f"Predicting for data: {new_data}")
    prediction = model.predict(X_new)
    predicted_class_index = prediction[0]
    predicted_class_name = iris.target_names[predicted_class_index]

    print(f"Predicted Class: {predicted_class_name} (Index: {predicted_class_index})")

if __name__ == "__main__":
    predict()

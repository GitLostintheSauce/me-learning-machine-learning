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
    import sys
    import os
    
    input_file = None
    if len(sys.argv) > 1:
        input_file = sys.argv[1]

    if input_file and os.path.exists(input_file):
        print(f"Reading data from {input_file}...")
        try:
            # specialized for the sample csv format
            X_new = pd.read_csv(input_file)
            print(f"Loaded {len(X_new)} samples.")
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        # Let's create a sample observation (a single flower).
        # Values correspond to: sepal length, sepal width, petal length, petal width
        # This sample is likely a 'setosa' (class 0).
        print("No input file provided (or file not found). Using hardcoded sample.")
        new_data = [[5.1, 3.5, 1.4, 0.2]]
        
        # We need the feature names to silence warnings from pandas/sklearn alignment
        iris = load_iris()
        feature_names = iris.feature_names
        X_new = pd.DataFrame(new_data, columns=feature_names)

    # 3. Make Prediction
    print("Making predictions...")
    predictions = model.predict(X_new)
    
    # Load iris for target names (class names)
    iris = load_iris()
    
    for i, pred_class_index in enumerate(predictions):
        predicted_class_name = iris.target_names[pred_class_index]
        # Get the row data for display
        row_data = X_new.iloc[i].values
        print(f"Row {i+1}: Data={row_data} -> Predicted Class: {predicted_class_name} (Index: {pred_class_index})")

if __name__ == "__main__":
    predict()

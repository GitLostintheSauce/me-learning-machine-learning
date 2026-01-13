# Simple Machine Learning Example: Iris Classification

This project demonstrates the fundamental concepts of machine learning using the classic Iris dataset. It includes scripts to train a model and make predictions.

## Key Concepts

1.  **Dataset (The Iris Data)**:
    *   **Features (X)**: Input data used to make predictions. Here: sepal length, sepal width, petal length, petal width.
    *   **Target (y)**: The output we want to predict. Here: the species of the flower (setosa, versicolor, virginica).

2.  **Training vs. Testing**:
    *   We split our data into two parts.
    *   **Training Set**: Used by the algorithm to learn the patterns.
    *   **Test Set**: Kept hidden from the model during training. Used to check if the model actually learned general patterns or just memorized the training data (overfitting).

3.  **Model (Decision Tree)**:
    *   We use a Decision Tree Classifier. It makes decisions by splitting data based on feature values (e.g., "If petal length < 2.5, it's setosa").

4.  **Serialization**:
    *   After training, we save the model to a file (`iris_model.joblib`). This allows us to use the trained model later (in other scripts or applications) without having to retrain it every time.

## File Structure

*   `train.py`: Loads data, trains the model, evaluates it, and saves it to a file.
*   `predict.py`: Loads the saved model and predicts the species for a new sample flower.
*   `requirements.txt`: Lists the Python libraries needed for this project.

## How to Run

### 1. Set up a Virtual Environment
It is best practice to use a virtual environment to manage dependencies locally.

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r ml_example/requirements.txt
```

### 3. Train the Model
Run the training script to build and save the model.
```bash
python ml_example/train.py
```
*Expected Output*: You should see accuracy metrics and a message that the model was saved.

### 4. Make a Prediction
Run the prediction script to see the model in action.
```bash
python ml_example/predict.py
```
*Expected Output*: The script will predict the class of a hardcoded sample flower.

---
**Note**: To deactivate the virtual environment when you are done, simply run:
```bash
deactivate
```

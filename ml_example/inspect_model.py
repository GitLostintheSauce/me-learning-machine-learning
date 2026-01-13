import joblib
from sklearn.tree import export_text

# Load the binary file
print("Loading 'iris_model.joblib'...")
model = joblib.load('ml_example/iris_model.joblib')

# Print what it actually is
print(f"\nType of object stored: {type(model)}")

# Since it's a Decision Tree, we can actually print the "Rules" it learned
print("\n--- The 'If Statements' inside the binary file ---")
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
tree_rules = export_text(model, feature_names=feature_names)
print(tree_rules)

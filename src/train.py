from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import mlflow
import requests
import os
import sys

mlflow.set_tracking_uri("http://34.170.230.68:5000")
mlflow.sklearn.autolog(
	max_tuning_runs=10,
	registered_model_name="iris-classifier"
)

data = pd.read_csv("data/data.csv")

train, test = train_test_split(data, test_size = 0.2, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'class_weight': [None, 'balanced']
}

with mlflow.start_run(run_name="Decision Tree Classifier Hyperparameter Tuning"):
	rf = DecisionTreeClassifier(random_state=42)
	grid_search = GridSearchCV(
    	rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
	)

	grid_search.fit(X_train, y_train)
    
	best_score = grid_search.score(X_test, y_test)
	print(f"Best parameters: {grid_search.best_params_}")
	print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
	print(f"Test score: {best_score:.3f}")

token = os.environ.get("GITHUB_PAT")
owner = "KSoham-dev"
repo = "mlops-assignments"
branch = "dev"

workflow_id = "ci.yml" 

url = f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches"

headers = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {token}",
}

data = {
    "ref": branch
}

print(f"Triggering workflow '{workflow_id}' on branch '{branch}'...")

try:
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 204:
        print(f"Successfully triggered GitHub Action workflow.")
    else:
        print(f"Failed to trigger workflow. Status: {response.status_code}, Response: {response.text}")
except Exception as e:
    print(f"Error triggering GitHub Action: {e}")
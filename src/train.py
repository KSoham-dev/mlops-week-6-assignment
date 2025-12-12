from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import mlflow
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = "iris-classifier-model-12"
artifact_location = "gs://mlops-course-clean-vista-473214-i6/mlflow-assets/iris-classifier-model"
registered_model_name = "iris-classifier"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.sklearn.autolog(
	max_tuning_runs=10,
	registered_model_name=registered_model_name
)

experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    logger.info(f"Creating new experiment '{experiment_name}'...")
    mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
else:
    logger.info(f"Using existing experiment '{experiment_name}'")
    mlflow.set_experiment(experiment_name)

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

def train_model():
    logger.info("Starting model training...")
    with mlflow.start_run(run_name="Decision Tree Classifier Hyperparameter Tuning"):
        logger.info("Initializing GridSearchCV...")
        rf = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
        )

        logger.info("Fitting model...")
        grid_search.fit(X_train, y_train)
        
        best_score = grid_search.score(X_test, y_test)
        
        # Log test metrics explicitly
        mlflow.log_metric("test_accuracy", best_score)
        mlflow.log_metric("cv_accuracy", grid_search.best_score_)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        logger.info(f"Test score: {best_score:.3f}")
        logger.info("Training completed successfully")

if __name__ == "__main__":
    train_model()
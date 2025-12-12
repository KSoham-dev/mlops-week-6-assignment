import mlflow
from tabulate import tabulate
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_NAME = "iris-classifier"
SAVE_PATH = "artifacts"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

def fetch_model():
    logger.info(f"Fetching latest model version for '{MODEL_NAME}'...")
    versions = client.search_model_versions(
        filter_string=f"name='{MODEL_NAME}'",
        order_by=["version_number DESC"],
        max_results=1
    )

    if not versions:
        logger.error(f"No versions found for model '{MODEL_NAME}'")
        raise ValueError(f"No versions found for model '{MODEL_NAME}'")

    latest_version = versions[0]
    logger.info(f"Found model version: {latest_version.version}, run_id: {latest_version.run_id}")

    run = client.get_run(latest_version.run_id)
    metrics = run.data.metrics
    header = ["Metric", "Value"]

    table_data = []
    for key, val in metrics.items():
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        table_data.append([key, val_str])

    metrics_table_string = tabulate(
            table_data, 
            headers=header, 
            tablefmt="github"
        )

    logger.info("Model metrics:")
    logger.info(f"\n{metrics_table_string}")

    logger.info(f"Downloading model artifacts to {SAVE_PATH}...")
    mlflow.artifacts.download_artifacts(
        run_id=latest_version.run_id,
        artifact_path="model",
        dst_path=SAVE_PATH
    )
    logger.info("Model artifacts downloaded")

    try:
        logger.info("Downloading confusion matrix...")
        mlflow.artifacts.download_artifacts(
                run_id=latest_version.run_id,
                artifact_path="training_confusion_matrix.png", # The *exact* name you logged in train.py
                dst_path=SAVE_PATH 
            )
        logger.info("Confusion matrix downloaded")
    except Exception as e:
        logger.warning(f"Failed to download confusion matrix: {e}")

    logger.info("Generating metrics.md file...")
    with open("metrics.md", "w") as f:
        f.write("### Metrics Table\n\n")
        f.write(f"{metrics_table_string}\n\n")
        f.write("### Confusion Matrix")
        f.write("![](./artifacts/training_confusion_matrix.png)")
    logger.info("metrics.md generated")
    
    # Return model version and metrics as dictionary
    result = {
        "version": str(latest_version.version),
        "metrics": metrics
    }
    logger.info(f"Model fetch completed: version={result['version']}")
    return result

if __name__ == "__main__":
    fetch_model()
import mlflow
from tabulate import tabulate

MODEL_NAME = "iris-classifier"
SAVE_PATH = "artifacts"


mlflow.set_tracking_uri("http://34.9.81.87:5000")
client = mlflow.tracking.MlflowClient()

versions = client.search_model_versions(
    filter_string=f"name='{MODEL_NAME}'",
    order_by=["version_number DESC"],
    max_results=1
)

latest_version = versions[0]

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

print(metrics_table_string)

mlflow.artifacts.download_artifacts(
    run_id=latest_version.run_id,
    artifact_path="model",
    dst_path=SAVE_PATH
)

mlflow.artifacts.download_artifacts(
        run_id=latest_version.run_id,
        artifact_path="training_confusion_matrix.png", # The *exact* name you logged in train.py
        dst_path=SAVE_PATH 
    )

with open("metrics.md", "w") as f:
    f.write("### Metrics Table\n\n")
    f.write(f"{metrics_table_string}\n\n")
    f.write("### Confusion Matrix")
    f.write("![](./artifacts/training_confusion_matrix.png)")


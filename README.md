| File / Directory | Purpose |
| --- | --- |
| `.github/workflows/ci.yml` | Runs automated tests and code checks on model train |
| `.github/workflows/cd.yml` | Automatically builds the Docker image and deploys the app to Kubernetes after CI passes. |
| `k8s/deployment.yaml` | A Kubernetes manifest that defines how to run the application's container as a service. |
| `Dockerfile` | A text file containing instructions to build the application into a portable Docker container image. |
| `src/train.py` | A Python script that trains the machine learning model and logs it to MLflow. |
| `src/fetch_model.py` | A utility script to download a specific registered model (e.g., "production") from MLflow. |
| `tests/test_model.py` | Contains unit tests to validate the model's logic and prevent regressions. |
| `main.py` | The FastAPI app that loads the model using `fetch_model.py` and serves predictions via an API. |

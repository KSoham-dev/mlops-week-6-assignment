from src.train import train_model as train_model_, registered_model_name
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.get_model import fetch_model as fetch_model_
import joblib, uvicorn, pandas as pd, os, mlflow, json, logging
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Config
MODEL_NAME = registered_model_name
MODEL_VERSION = "1.0.0"
API_VERSION = "1.0.0"
METRICS_CACHE = "artifacts/metrics_cache.json"

if MLFLOW_URI := os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(MLFLOW_URI)

_model = _version = _metrics = None

def _save_cache(version: str, metrics: Dict):
    try:
        os.makedirs("artifacts", exist_ok=True)
        with open(METRICS_CACHE, "w") as f:
            json.dump({"version": version, "metrics": metrics, "timestamp": datetime.now().isoformat()}, f)
        logger.debug(f"Saved metrics cache: version={version}")
    except Exception as e:
        logger.warning(f"Failed to save metrics cache: {e}")

def _load_cache() -> tuple:
    try:
        if os.path.exists(METRICS_CACHE):
            with open(METRICS_CACHE) as f:
                d = json.load(f)
                logger.debug(f"Loaded metrics cache: version={d.get('version')}")
                return d.get("version"), d.get("metrics", {})
    except Exception as e:
        logger.warning(f"Failed to load metrics cache: {e}")
    return None, None

def load_model():
    global _model, _version, _metrics
    if _model:
        logger.debug("Returning cached model")
        return _model, _version, _metrics
    
    logger.info("Loading model...")
    model_path = "artifacts/model/model.pkl"
    if os.path.exists(model_path):
        _model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        try:
            r = fetch_model_()
            _version = r.get("version", MODEL_VERSION) if r else MODEL_VERSION
            _metrics = r.get("metrics", {}) if r else {}
            _save_cache(_version, _metrics)
            logger.info(f"Model metadata fetched: version={_version}")
        except Exception as e:
            logger.warning(f"Failed to fetch model metadata from MLflow: {e}")
            _version, _metrics = _load_cache()
            if not _version:
                _version, _metrics = MODEL_VERSION, {}
        return _model, _version, _metrics
    
    try:
        logger.info("Attempting to fetch model from MLflow...")
        r = fetch_model_()
        if r and os.path.exists(model_path):
            _model = joblib.load(model_path)
            _version, _metrics = r.get("version", MODEL_VERSION), r.get("metrics", {})
            _save_cache(_version, _metrics)
            logger.info(f"Model loaded successfully: version={_version}")
            return _model, _version, _metrics
    except Exception as e:
        logger.error(f"Failed to fetch and load model: {e}")
    
    _version, _metrics = _load_cache()
    if not _version:
        logger.error("Model not found and no cache available")
        raise FileNotFoundError("Model not found")
    logger.error("Model file not found")
    raise FileNotFoundError("Model not found")

def clear_cache():
    global _model, _version, _metrics
    _model = _version = _metrics = None
    logger.info("Model cache cleared")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("Application startup - loading cached metrics")
    v, m = _load_cache()
    if v:
        global _version, _metrics
        _version, _metrics = v, m
        logger.info(f"Loaded cached model version: {v}")
    yield
    logger.info("Application shutdown")

app = FastAPI(title="Iris Classifier API", version=API_VERSION, lifespan=lifespan)

class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Prediction(BaseModel):
    prediction: str
    confidence: float = None
    timestamp: str = None
    model_version: str = None

class Training(BaseModel):
    status: str
    message: str
    timestamp: str
    experiment_name: str = None

class ModelInfo(BaseModel):
    model_name: str
    version: str
    metrics: Dict[str, Any]
    timestamp: str

class Batch(BaseModel):
    predictions: List[str]
    count: int
    timestamp: str

@app.get("/health")
def health():
    loaded = os.path.exists("artifacts/model/model.pkl")
    logger.debug(f"Health check: model_loaded={loaded}")
    return {"status": "healthy" if loaded else "degraded", "model_loaded": loaded, "timestamp": datetime.now().isoformat(), "version": API_VERSION}

@app.get("/info")
def info():
    return {"api_name": "Iris Classifier API", "version": API_VERSION}

@app.get("/model_info", response_model=ModelInfo)
def model_info():
    logger.info("Model info request received")
    try:
        _, v, m = load_model()
        return {"model_name": MODEL_NAME, "version": v, "metrics": m or {}, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Unable to fetch model info: {e}")
        raise HTTPException(503, f"Unable to fetch model: {e}")

@app.post("/train", response_model=Training)
def train():
    logger.info("Training request received")
    try:
        train_model_()
        clear_cache()
        logger.info("Training completed successfully")
        return {"status": "success", "message": "Training completed", "timestamp": datetime.now().isoformat(), "experiment_name": "iris-classifier-model-12"}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(500, f"Training failed: {e}")

@app.post("/predict", response_model=Prediction)
def predict(sample: Iris):
    logger.info(f"Prediction request received: {sample.model_dump()}")
    try:
        model, ver, _ = load_model()
    except Exception as e:
        logger.error(f"Model unavailable for prediction: {e}")
        raise HTTPException(503, f"Model unavailable: {e}")
    
    df = pd.DataFrame([sample.model_dump()])
    df = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    
    try:
        pred = model.predict(df)[0]
        conf = float(max(model.predict_proba(df)[0]))
        logger.info(f"Prediction successful: {pred} (confidence: {conf:.4f})")
        return {"prediction": str(pred), "confidence": round(conf, 4), "timestamp": datetime.now().isoformat(), "model_version": ver}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(500, f"Prediction failed: {e}")

@app.post("/batch_predict", response_model=Batch)
def batch_predict(samples: List[Iris]):
    logger.info(f"Batch prediction request received: {len(samples)} samples")
    if not samples or len(samples) > 100:
        logger.warning(f"Invalid batch size: {len(samples) if samples else 0}")
        raise HTTPException(400, "Invalid sample count")
    
    try:
        model, _, _ = load_model()
    except Exception as e:
        logger.error(f"Model unavailable for batch prediction: {e}")
        raise HTTPException(503, f"Model unavailable: {e}")
    
    try:
        df = pd.DataFrame([s.model_dump() for s in samples])
        df = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        preds = model.predict(df)
        logger.info(f"Batch prediction successful: {len(preds)} predictions made")
        return {"predictions": [str(p) for p in preds], "count": len(preds), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(500, f"Batch prediction failed: {e}")

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)

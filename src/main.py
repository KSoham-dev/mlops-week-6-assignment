from src.train import train_model as train_model_
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.get_model import fetch_model as fetch_model_
import joblib
import pandas as pd


app = FastAPI(title="iris-classifier")

class IrisSample(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: str


@app.get("/train_model")
def train_model():
    train_model_()
    return {"Model ran successfully"}


@app.get("/fetch_model")
def fetch_model():
    fetch_model_()
    return {"Model fetched successfully"}

@app.post("/predict", response_model=PredictionResponse, tags=["Inference/Deployment"])
def predict(sample: IrisSample):
    try:
        model_uri = "artifacts/model/model.pkl"
        
        loaded_model = joblib.load(model_uri)    
        
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Check MLflow connection/registry stage."
        )

    input_data = pd.DataFrame([sample.model_dump()])
    
    input_data = input_data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

    try:
        prediction = loaded_model.predict(input_data)[0]
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e.__class__.__name__}")


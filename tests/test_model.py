import pytest
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

@pytest.fixture()
def data():
    data = pd.read_csv("./data.csv")
    return data

@pytest.fixture()
def trained_model():
    model = joblib.load("./artifacts/model/model.pkl") 
    return model

def test_data_no_null_values(data):
    assert data.isnull().sum().sum() == 0

def test_model_accuracy(trained_model, data):
    X = data
    y = data.species
    X.drop("species", axis=1, inplace=True)
    y_pred = trained_model.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.90

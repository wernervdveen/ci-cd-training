import os
import uvicorn
import numpy as np
import pandas as pd
from patsy import dmatrix
import joblib
from config import Config
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


def load_model(model_path: str):
    return joblib.load(model_path)


def add_dummy_data(df: pd.DataFrame):
    """Add dummy rows so that patsy can create features similar to the train dataset"""
    rows = {
        "City": ["Bangalore", "New Delhi", "Prune"],
        "Gender": ["Male", "Female", "Female"],
        "EverBenched": ["Yes", "Yes", "No"],
        "PaymentTier": [0, 0, 0],
        "Age": [0, 0, 0],
        "ExperienceInCurrentDomain": [0, 0, 0],
    }
    dummy_df = pd.DataFrame(rows)
    return pd.concat([df, dummy_df])


def rename_columns(X: pd.DataFrame):
    X.columns = X.columns.str.replace("[", "_", regex=True).str.replace(
        "]", "", regex=True
    )
    return X


def transform_data(df: pd.DataFrame, config: Config):
    """Transform the data"""
    dummy_df = add_dummy_data(df)
    feature_str = " + ".join(config.FEATURES)
    dummy_X = dmatrix(f"{feature_str} - 1", dummy_df, return_type="dataframe")
    dummy_X = rename_columns(dummy_X)
    return dummy_X.iloc[0, :].values.reshape(1, -1)


@app.post("/predict")
async def predict(request: Request) -> np.ndarray:
    """Transform the data then make predictions"""
    data = pd.DataFrame.from_dict(request)
    df = transform_data(data, config)
    result = model.predict(df)
    response_dict = dict()
    response_dict["prediction"] = result.tolist()
    return JSONResponse(content=response_dict)

if __name__ == "__main__":
    config = Config()
    model = load_model(f"{config.PROJECT_ROOT}/{config.MODEL_PATH}/model.pickle")
    uvicorn.run(app, host="0.0.0.0", port=os.environ.get('PORT', '3000'))
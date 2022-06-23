import pandas as pd
import joblib


def load_raw_data(raw_path: str):
    data = pd.read_csv(raw_path)
    return data


def load_processed_data(processed_path: str):
    return pd.read_csv(processed_path)


def load_model(model_path: str):
    return joblib.load(model_path)

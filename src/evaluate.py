import warnings
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from config import Config
from src.helpers import load_processed_data, load_model
warnings.filterwarnings(action="ignore")



def predict(model: XGBClassifier, X_test: pd.DataFrame):
    return model.predict(X_test)


def evaluate(config: Config):

    # Load data and model
    X_test = load_processed_data(f"{config.PROJECT_ROOT}/{config.DATA_PROCESSED_PATH}/X_test.csv")
    y_test = load_processed_data(f"{config.PROJECT_ROOT}/{config.DATA_PROCESSED_PATH}/y_test.csv")

    model = load_model(f"{config.PROJECT_ROOT}/{config.MODEL_PATH}/model.pickle")

    # Get predictions
    prediction = predict(model, X_test)

    # Get metrics
    f1 = f1_score(y_test, prediction)
    print(f"F1 Score of this model is {f1}.")

    accuracy = accuracy_score(y_test, prediction)
    print(f"Accuracy Score of this model is {accuracy}.")


if __name__ == "__main__":
    evaluate()

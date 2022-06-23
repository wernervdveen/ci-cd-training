import warnings

from functools import partial
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.helpers import load_processed_data
from config import Config

warnings.filterwarnings(action="ignore")


def get_objective(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    space: dict,
):

    model = XGBClassifier(
        use_label_encoder=False,
        objective="binary:logistic",
        n_estimators=space["n_estimators"],
        max_depth=int(space["max_depth"]),
        gamma=space["gamma"],
        reg_alpha=int(space["reg_alpha"]),
        min_child_weight=int(space["min_child_weight"]),
        colsample_bytree=int(space["colsample_bytree"]),
    )

    evaluation = [(X_train, y_train), (X_test, y_test)]

    model.fit(
        X_train,
        y_train,
        eval_set=evaluation,
        eval_metric="auc",
        early_stopping_rounds=10,
    )
    prediction = model.predict(X_test.values)
    accuracy = accuracy_score(y_test, prediction)
    print("SCORE:", accuracy)
    return {"loss": -accuracy, "status": STATUS_OK, "model": model}


def optimize(objective: Callable, space: dict):
    trials = Trials()
    best_hyperparams = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
    )
    print("The best hyperparameters are : ", "\n")
    print(best_hyperparams)
    best_model = trials.results[
        np.argmin([r["loss"] for r in trials.results])
    ]["model"]
    return best_model


def train(config: Config):
    """Function to train the model"""

    X_train, X_test, y_train, y_test, = [
        load_processed_data(
            f"{config.PROJECT_ROOT}/{config.DATA_PROCESSED_PATH}/{df}"
            f".csv")
        for df in ["X_train", "X_test", "y_train", "y_test"]
    ]

    # Define space
    space = {
        "max_depth": hp.quniform("max_depth", 3, 18, 1),
        "gamma": hp.uniform("gamma", 1, 9),
        "reg_alpha": hp.quniform("reg_alpha", 40, 180, 1),
        "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        "colsample_bytree": hp.uniform(
            "colsample_bytree", 0.5, 1
        ),
        "min_child_weight": hp.quniform(
            "min_child_weight", 0, 10, 1
        ),
        "n_estimators": 150,
        "seed": 0,
    }
    objective = partial(
        get_objective, X_train, y_train, X_test, y_test
    )

    # Find best model
    best_model = optimize(objective, space)

    # Save model
    joblib.dump(best_model,
                f"{config.PROJECT_ROOT}/{config.MODEL_PATH}/model.pickle")


if __name__ == "__main__":
    train()

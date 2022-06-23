import pandas as pd
from sklearn.model_selection import train_test_split
from patsy import dmatrices

from src.helpers import load_raw_data
from config import Config


def get_features(target: str, features: list, data: pd.DataFrame):
    feature_str = " + ".join(features)
    y, X = dmatrices(
        f"{target} ~ {feature_str} - 1", data=data, return_type="dataframe"
    )
    return y, X


def rename_columns(X: pd.DataFrame):
    X.columns = X.columns.str.replace("[", "_", regex=True).str.replace(
        "]", "", regex=True
    )
    return X


def process_data(config: Config):
    """Function to process the data"""

    data = load_raw_data(f"{config.PROJECT_ROOT}/{config.DATA_RAW_PATH}/employees.csv")

    y, X = get_features(config.TARGET, config.FEATURES, data)

    X = rename_columns(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7
    )

    # Save data
    X_train.to_csv(f"{config.PROJECT_ROOT}/{config.DATA_PROCESSED_PATH}/X_train.csv", index=False)
    X_test.to_csv(f"{config.PROJECT_ROOT}/{config.DATA_PROCESSED_PATH}/X_test.csv", index=False)
    y_train.to_csv(f"{config.PROJECT_ROOT}/{config.DATA_PROCESSED_PATH}/y_train.csv", index=False)
    y_test.to_csv(f"{config.PROJECT_ROOT}/{config.DATA_PROCESSED_PATH}/y_test.csv", index=False)


if __name__ == "__main__":
    process_data()

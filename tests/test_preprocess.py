import pandas as pd
from src.preprocess import get_features, rename_columns


def test_get_features():
    df_input = pd.DataFrame(
        {
            "Education": ["Bachelors", "Masters"],
            "City": ["Bangalore", "Prune"],
            "PaymentTier": [2.0, 3.0],
            "Age": [30.0, 21.0],
            "Gender": ["Male", "Female"],
            "EverBenched": ["No", "Yes"],
            "ExperienceInCurrentDomain": [2.0, 3.0],
            "LeaveOrNot": [0.0, 1.0],
        }
    )
    X_expected = pd.DataFrame(
        {
            "City[Bangalore]": [1.0, 0.0],
            "City[Prune]": [0.0, 1.0],
            "Gender[T.Male]": [1.0, 0.0],
            "EverBenched[T.Yes]": [0.0, 1.0],
            "PaymentTier": [2.0, 3.0],
            "Age": [30.0, 21.0],
            "ExperienceInCurrentDomain": [2.0, 3.0],
        }
    )
    y_expected = pd.DataFrame(
        {
            "LeaveOrNot": [0.0, 1.0]
        }

    )
    features = [
        "City",
        "PaymentTier",
        "Age",
        "Gender",
        "EverBenched",
        "ExperienceInCurrentDomain",
    ]
    target = "LeaveOrNot"
    y_out, X_out = get_features(target, features, df_input)
    pd.testing.assert_frame_equal(y_out, y_expected)
    pd.testing.assert_frame_equal(X_out, X_expected)


def test_rename_columns():
    X_input = pd.DataFrame(
        {
            "City[Bangalore]": [1.0, 0.0],
            "City[Prune]": [0.0, 1.0],
            "Gender[T.Male]": [1.0, 0.0],
            "EverBenched[T.Yes]": [0.0, 1.0],
            "PaymentTier": [2.0, 3.0],
            "Age": [30.0, 21.0],
            "ExperienceInCurrentDomain": [2.0, 3.0],
        }
    )
    X_out = rename_columns(X_input)
    assert list(X_out.columns) == [
        "City_Bangalore",
        "City_Prune",
        "Gender_T.Male",
        "EverBenched_T.Yes",
        "PaymentTier",
        "Age",
        "ExperienceInCurrentDomain",
    ]
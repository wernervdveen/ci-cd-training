import os


class Config:
    """Class which contains our configuration."""

    def __init__(
        self,
        project_root: str = None,
    ):
        if project_root:
            self.PROJECT_ROOT = project_root
        else:
            self.PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "../"))

        if not os.path.exists(self.PROJECT_ROOT):
            raise FileNotFoundError("The path given as project root does not exist")
        if not os.path.isdir(self.PROJECT_ROOT):
            raise NotADirectoryError(
                "The path given is not a directory, but a file path"
            )

    DATA_RAW_PATH = "/data/raw"
    DATA_PROCESSED_PATH = "/data/processed"
    MODEL_PATH = "/models"

    TARGET = "LeaveOrNot"
    FEATURES = [
        "City",
        "PaymentTier",
        "Age",
        "Gender",
        "EverBenched",
        "ExperienceInCurrentDomain"
    ]
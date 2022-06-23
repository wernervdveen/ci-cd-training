from src.evaluate import evaluate
from src.preprocess import process_data
from src.train import train
from config import Config


def main(config):
    process_data(config)
    train(config)
    evaluate(config)


if __name__ == "__main__":
    config = Config()
    main(config)

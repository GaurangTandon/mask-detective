from .train import train_model
from .predict import predictor
from .dataset import get_dataset


def main():
    # TODO: get dataset
    PATH = "model.h5"
    train_model(PATH)
    predictions = predictor(PATH)
    return predictions
import configparser
from src.model_fine_tuning.train import train

config = configparser.ConfigParser()
config.read("src/model_fine_tuning/fine_tuning_config.ini")


if __name__ == "__main__":
    train(config)
import configparser
from services.trainer.trainer import Trainer

config = configparser.ConfigParser()
config.read("services/trainer/fine_tuning_config.ini")


if __name__ == "__main__":
    model_trainer = Trainer(config)
    model_trainer.train()
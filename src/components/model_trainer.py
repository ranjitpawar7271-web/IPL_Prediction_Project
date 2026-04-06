import os
import sys
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact
from src.exception import CustomException
from src.logger import logging


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self, train_path, test_path):
        logging.info("Starting model training")

        try:
            train_arr = np.load(train_path)
            test_arr = np.load(test_path)

            X_train, y_train_raw = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test_raw = test_arr[:, :-1], test_arr[:, -1]

            # 🔥 Convert to binary classification (team1 vs team2)
            y_train = (y_train_raw == X_train[:, 0]).astype(int)
            y_test = (y_test_raw == X_test[:, 0]).astype(int)

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            logging.info("Model training completed")

            os.makedirs(self.config.model_trainer_dir, exist_ok=True)

            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(model, f)

            logging.info("Model saved")

            return ModelTrainerArtifact(
                model_file_path=self.config.trained_model_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
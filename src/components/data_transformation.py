import os
import sys
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact
from src.exception import CustomException
from src.logger import logging
from src.constant.training_pipeline import TARGET_COLUMN


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Starting data transformation")

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            train_df.drop(columns=["match_id"], inplace=True)
            test_df.drop(columns=["match_id"], inplace=True)

            
            train_df[TARGET_COLUMN] = (train_df[TARGET_COLUMN] == train_df["team1"]).astype(int)
            test_df[TARGET_COLUMN] = (test_df[TARGET_COLUMN] == test_df["team1"]).astype(int)

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            encoders = {}

            for col in X_train.columns:
                le = LabelEncoder()

                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))

                encoders[col] = le

            

            train_arr = np.c_[X_train, y_train]
            test_arr = np.c_[X_test, y_test]

            os.makedirs(self.config.data_transformation_dir, exist_ok=True)

            np.save(self.config.transformed_train_file_path, train_arr)
            np.save(self.config.transformed_test_file_path, test_arr)

            with open(self.config.preprocessor_object_file_path, "wb") as f:
                pickle.dump(encoders, f)

            logging.info("Data transformation completed")

            return DataTransformationArtifact(
                train_file_path=self.config.transformed_train_file_path,
                test_file_path=self.config.transformed_test_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
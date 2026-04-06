import os
import sys
import pandas as pd

from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact
from src.exception import CustomException
from src.logger import logging
from src.constant.training_pipeline import REQUIRED_COLUMNS


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_columns(self, df: pd.DataFrame) -> bool:
        try:
            logging.info("Validating required columns")

            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    logging.error(f"Missing column: {col}")
                    return False

            return True

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self, train_path: str) -> DataValidationArtifact:
        logging.info("Starting data validation")

        try:
            df = pd.read_csv(train_path)

            status = self.validate_columns(df)

            os.makedirs(self.config.data_validation_dir, exist_ok=True)

            if status:
                df.to_csv(self.config.valid_data_file_path, index=False)
                logging.info("Data validation passed")
            else:
                df.to_csv(self.config.invalid_data_file_path, index=False)
                logging.info("Data validation failed")

            return DataValidationArtifact(validation_status=status)

        except Exception as e:
            raise CustomException(e, sys)


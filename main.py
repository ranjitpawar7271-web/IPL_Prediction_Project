from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys


if __name__ == "__main__":
    try:
        logging.info("Pipeline started")

        config = ConfigurationManager()
        training_pipeline_config = config.get_training_pipeline_config()

       
        data_ingestion_config = config.get_data_ingestion_config(
            training_pipeline_config
        )

        data_ingestion = DataIngestion(data_ingestion_config)

        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        print(data_ingestion_artifact.train_file_path)
        print(data_ingestion_artifact.test_file_path)

       
        data_validation_config = config.get_data_validation_config(
            training_pipeline_config
        )

        data_validation = DataValidation(data_validation_config)

        validation_artifact = data_validation.initiate_data_validation(
            data_ingestion_artifact.train_file_path
        )

        print("Validation Status:", validation_artifact.validation_status)

       
        data_transformation_config = config.get_data_transformation_config(
            training_pipeline_config
        )

        data_transformation = DataTransformation(data_transformation_config)

        transformation_artifact = data_transformation.initiate_data_transformation(
            data_ingestion_artifact.train_file_path,
            data_ingestion_artifact.test_file_path
        )

        print("Transformation Done")

        model_trainer_config = config.get_model_trainer_config(
                    training_pipeline_config
            )       

        model_trainer = ModelTrainer(model_trainer_config)

        model_artifact = model_trainer.initiate_model_trainer(
            transformation_artifact.train_file_path,
            transformation_artifact.test_file_path
        )

        print("Model Training Done")

    except Exception as e:
        raise CustomException(e, sys)
import os
from src.constant.training_pipeline import *

class TrainingPipelineConfig:
    def __init__(self):
        self.artifact_dir = os.path.join(
            ARTIFACT_DIR, CURRENT_TIME_STAMP
        )

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            "data_ingestion"
        )

        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            FILE_NAME
        )

        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            TRAIN_FILE_NAME
        )

        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            TEST_FILE_NAME
        )

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            "data_validation"
        )

        self.valid_data_file_path = os.path.join(
            self.data_validation_dir,
            "valid.csv"
        )

        self.invalid_data_file_path = os.path.join(
            self.data_validation_dir,
            "invalid.csv"
        )
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            "data_transformation"
        )

        self.transformed_train_file_path = os.path.join(
            self.data_transformation_dir,
            "train.npy"
        )

        self.transformed_test_file_path = os.path.join(
            self.data_transformation_dir,
            "test.npy"
        )

        self.preprocessor_object_file_path = os.path.join(
            self.data_transformation_dir,
            "preprocessor.pkl"
        )

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):

        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            "model_trainer"
        )

        self.trained_model_file_path = os.path.join(
            self.model_trainer_dir,
            "model.pkl"
        )
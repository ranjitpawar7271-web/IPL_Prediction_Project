from src.entity.config_entity import DataValidationConfig
from src.entity.config_entity import DataTransformationConfig
from src.entity.config_entity import ModelTrainerConfig
from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig
)

class ConfigurationManager:

    def get_training_pipeline_config(self):
        return TrainingPipelineConfig()

    def get_data_ingestion_config(self, training_pipeline_config):
        return DataIngestionConfig(training_pipeline_config)

    def get_data_validation_config(self, training_pipeline_config):
        return DataValidationConfig(training_pipeline_config)

    def get_data_transformation_config(self, training_pipeline_config):
        return DataTransformationConfig(training_pipeline_config)

    def get_model_trainer_config(self, training_pipeline_config):
        return ModelTrainerConfig(training_pipeline_config)
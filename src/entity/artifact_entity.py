
class DataIngestionArtifact:
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path

class DataValidationArtifact:
    def __init__(self, validation_status: bool):
        self.validation_status = validation_status

class DataTransformationArtifact:
    def __init__(self, train_file_path, test_file_path):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
class ModelTrainerArtifact:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
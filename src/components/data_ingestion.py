import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import CustomException
from src.logger import logging


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered data ingestion method")

        try:
            # Load ball-by-ball dataset
            df = pd.read_csv("data/raw/IPL.csv", low_memory=False)
            logging.info("Raw ball-by-ball data loaded")

           
            logging.info("Converting ball-by-ball to match-level data")

            df_match = df.groupby("match_id").agg({
                "batting_team": lambda x: list(x.unique())[0],
                "bowling_team": lambda x: list(x.unique())[0],
                "toss_winner": "first",
                "toss_decision": "first",
                "venue": "first",
                "match_won_by": "first"
            }).reset_index()

            df_match.rename(columns={
                "batting_team": "team1",
                "bowling_team": "team2",
                "match_won_by": "winner"
            }, inplace=True)

            logging.info("Match-level dataset created")

            
            os.makedirs(self.config.data_ingestion_dir, exist_ok=True)

          
            df_match.to_csv(self.config.feature_store_file_path, index=False)

            
            train_set, test_set = train_test_split(
                df_match, test_size=0.2, random_state=42
            )

            train_set.to_csv(self.config.training_file_path, index=False)
            test_set.to_csv(self.config.testing_file_path, index=False)

            logging.info("Train-test split completed")

            return DataIngestionArtifact(
                train_file_path=self.config.training_file_path,
                test_file_path=self.config.testing_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
import os
from datetime import datetime
PIPELINE_NAME = "IPL_PREDICTION"
ARTIFACT_DIR = "artifacts"
FILE_NAME = "IPL.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TARGET_COLUMN = "winner"
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

REQUIRED_COLUMNS = [
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "winner",
    "venue"
]
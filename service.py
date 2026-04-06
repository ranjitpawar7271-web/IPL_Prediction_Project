import bentoml
import pandas as pd
import pickle
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

model = bentoml.sklearn.load_model("ipl_model:latest")

artifact_dir = "artifacts"
latest_folder = sorted(os.listdir(artifact_dir))[-1]

preprocessor_path = os.path.join(
    artifact_dir,
    latest_folder,
    "data_transformation",
    "preprocessor.pkl"
)

with open(preprocessor_path, "rb") as f:
    encoders = pickle.load(f)

CURRENT_TEAMS = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Lucknow Super Giants",
    "Gujarat Titans"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

svc = bentoml.Service("ipl_service")

svc.mount_asgi_app(app, path="/")

@app.get("/metadata")
def metadata():
    try:
        all_teams = encoders["team1"].classes_.tolist()
        filtered_teams = [t for t in all_teams if t in CURRENT_TEAMS]

        raw_venues = encoders["venue"].classes_.tolist()
        cleaned_venues = sorted(list(set(raw_venues)))
        cleaned_venues.sort()

        return {
            "teams": filtered_teams,
            "venues": cleaned_venues,
            "toss_decision": encoders["toss_decision"].classes_.tolist()
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
def predict(input_data: dict):
    try:
        if "input_data" in input_data:
            data = input_data["input_data"]["data"]
        else:
            data = input_data

        venue_input = data.get("venue", "")
        possible = [v for v in encoders["venue"].classes_ if v.startswith(venue_input)]
        if possible:
            data["venue"] = possible[0]

        df = pd.DataFrame([data])

        for col in df.columns:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])

        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0]

        confidence = max(proba)

        if int(prediction) == 1:
            winner_team = data["team1"]
        else:
            winner_team = data["team2"]

        return {
            "prediction": winner_team,
            "confidence": round(confidence * 100, 2),
            "status": "success"
        }

        if int(prediction) == 1:
            winner_team = data["team1"]
        else:
            winner_team = data["team2"]

        return {
            "prediction": winner_team,
            "status": "success"
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }
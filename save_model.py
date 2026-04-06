import bentoml
import pickle


with open("artifacts/2026-04-04-19-29-26/model_trainer/model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully")


bentoml.sklearn.save_model("ipl_model", model)

print("Model saved into BentoML")
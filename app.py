from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load models
clf = joblib.load("injury_classifier.joblib")
reg = joblib.load("recovery_regressor.joblib")
scaler = joblib.load("scaler.joblib")

class Player(BaseModel):
    Player_Age: float
    Player_Weight: float
    Player_Height: float
    Previous_Injuries: int
    Training_Intensity: float

@app.post("/predict")
def predict(player: Player):
    df = pd.DataFrame([player.dict()])
    X_scaled = scaler.transform(df)
    injury_prob = float(clf.predict_proba(X_scaled)[:, 1][0])
    injury_pred = int(clf.predict(X_scaled)[0])
    recovery_pred = float(reg.predict(X_scaled)[0])
    return {
        "injury_probability": injury_prob,
        "predicted_injury": injury_pred,
        "predicted_recovery_time": recovery_pred
    }


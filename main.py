import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import librosa
import io
import base64

app = FastAPI()

# Load model
model = joblib.load("model.pkl")


class AudioRequest(BaseModel):
    audio_base64: str


def extract_features(file_bytes):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    features = np.hstack([
        mfccs_mean,
        zcr,
        spectral_centroid,
        spectral_flatness
    ])

    return features.reshape(1, -1)


@app.get("/")
def home():
    return {
        "status": "running",
        "message": "AI Voice Detector API is live"
    }


@app.post("/predict")
def predict(data: AudioRequest):
    try:
        audio_bytes = base64.b64decode(data.audio_base64)
        features = extract_features(audio_bytes)
        prediction = model.predict(features)

        return {
            "prediction": "AI" if prediction[0] == 1 else "Human"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

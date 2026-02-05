from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import librosa
import io
import base64

app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")

# Root endpoint
@app.get("/")
def home():
    return {
        "status": "running",
        "message": "AI Voice Detector API is live"
    }

# Request body format
class AudioRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str

# Feature extraction
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

# Prediction endpoint
@app.post("/detect")
def detect_voice(
    request: AudioRequest,
    x_api_key: str = Header(...)
):
    # Simple API key check
    if x_api_key != "test12345":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        features = extract_features(audio_bytes)

        prediction = model.predict(features)[0]
        confidence = float(np.max(model.predict_proba(features)))

        return {
            "prediction": "Human" if prediction == 1 else "AI",
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

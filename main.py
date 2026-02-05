from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import librosa
import io
import base64

app = FastAPI(title="AI Voice Detector")

# Load trained model
model = joblib.load("model.pkl")


# ===== Request Body =====
class AudioRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str


# ===== Health Check =====
@app.get("/")
def home():
    return {"status": "running", "message": "AI Voice Detector API is live"}


# ===== Predict Endpoint =====
@app.post("/detect")
def detect_voice(
    request: AudioRequest,
    x_api_key: str = Header(...)
):
    # Validate API key
    if x_api_key != "test12345":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(request.audio_base64)

        # Extract features
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

        features = np.hstack([mfcc, zcr, spectral_flatness]).reshape(1, -1)

        # Predict using model
        prediction = model.predict(features)[0]
        confidence = float(np.max(model.predict_proba(features)))

        # Return final result
        return {
            "language": request.language,
            "audio_format": request.audio_format,
            "prediction": "AI Generated" if prediction == 1 else "Human",
            "confidence": round(confidence, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

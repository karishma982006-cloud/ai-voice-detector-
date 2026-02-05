from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import librosa
import io
import base64

app = FastAPI()

# 🔐 Your Secret API Key
API_KEY = "test12345"

# Load trained model
model = joblib.load("model.pkl")


# ==========================
# Request Body Schema
# ==========================
class AudioRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str


# ==========================
# Feature Extraction
# ==========================
def extract_features(file_bytes):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches)

    features = np.hstack([mfcc, zcr, spectral_flatness, pitch])
    return features.reshape(1, -1)


# ==========================
# Detect Endpoint
# ==========================
@app.post("/detect")
async def detect_audio(
    request: AudioRequest,
    x_api_key: str = Header(...)
):
    # 🔐 API Key Validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # Decode Base64
        audio_bytes = base64.b64decode(request.audio_base64)

        # Extract Features
        features = extract_features(audio_bytes)

        # Predict
        prediction = model.predict(features)[0]
        probs = model.predict_proba(features)[0]

        label = "AI Generated" if prediction == 1 else "Human"
        confidence = float(np.max(probs))

        return {
            "prediction": label,
            "confidence": round(confidence, 3)
        }

    except Exception as e:
        return {"error": str(e)}

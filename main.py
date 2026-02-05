from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import librosa
import io
import base64

app = FastAPI(title="AI Voice Detector")

# Load the trained model
model = joblib.load("model.pkl")


# ===== REQUEST BODY =====
class AudioRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str


# ===== FEATURE EXTRACTION =====
def extract_features(audio_bytes):
    # Convert bytes into audio array
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # 13 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

    # Spectral Flatness
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    # Pitch (IMPORTANT)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches)

    # Combine into 16 features
    features = np.hstack([
        mfccs_mean,
        zcr,
        spectral_flatness,
        pitch
    ])

    # Return shape (1, 16)
    return features.reshape(1, -1)


# ===== HEALTH CHECK =====
@app.get("/")
def home():
    return {"status": "running", "message": "AI Voice Detector API is live"}


# ===== PREDICTION ENDPOINT =====
@app.post("/detect")
def detect_voice(
    request: AudioRequest,
    x_api_key: str = Header(...)
):

    # Validate API key
    if x_api_key != "test12345":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # Decode Base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)

        # Extract features
        features = extract_features(audio_bytes)

        # Predict using model
        prediction = model.predict(features)[0]
        confidence = float(np.max(model.predict_proba(features)))

        # Return JSON response
        return {
            "language": request.language,
            "audio_format": request.audio_format,
            "prediction": "AI Generated" if prediction == 1 else "Human",
            "confidence": round(confidence, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

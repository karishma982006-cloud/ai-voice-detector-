import librosa
import numpy as np
import io

# Used for BOTH training and prediction
def extract_features(y, sr):
    # 13 MFCC
    mfcc = np.mean(
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13),
        axis=1
    )

    # ZCR
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(y=y)
    )

    # Spectral Flatness
    spectral_flatness = np.mean(
        librosa.feature.spectral_flatness(y=y)
    )

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches)

    features = np.hstack([
        mfcc,
        zcr,
        spectral_flatness,
        pitch
    ])

    return features


def extract_features_from_path(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return extract_features(y, sr)


def extract_features_from_bytes(file_bytes):
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=None)
    return extract_features(y, sr)

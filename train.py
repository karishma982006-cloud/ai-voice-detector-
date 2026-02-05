import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from audio_utils import extract_features_from_path


def load_dataset(data_path="data"):
    X, y = [], []

    for label, category in enumerate(["human", "ai"]):
        folder = os.path.join(data_path, category)

        for file in os.listdir(folder):
            if file.endswith(".wav"):
                features = extract_features_from_path(
                    os.path.join(folder, file)
                )
                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)


def train():
    print("📂 Loading dataset...")
    X, y = load_dataset()

    print(f"✅ Total samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training model...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"🎯 Model Accuracy: {acc*100:.2f}%")

    joblib.dump(model, "model.pkl")
    print("💾 Model saved as model.pkl")


if __name__ == "__main__":
    train()

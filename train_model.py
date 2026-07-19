
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split


SIGNS = [
    "help",
    "doctor",
    "medicine",
    "pharmacy",
    "emergency",
    "pain",
    "hospital_ward",
    "swelling",
    "blood_pressure",
    "bleeding"
]
DATA_DIRECTORY = Path("data")
MODEL_DIRECTORY = Path("models")
MODEL_PATH = MODEL_DIRECTORY / "eshara_rf.joblib"

EXPECTED_SHAPE = (30, 132)
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_dataset():
    features = []
    labels = []

    for label_index, sign_name in enumerate(SIGNS):
        sign_directory = DATA_DIRECTORY / sign_name
        sample_files = sorted(
            sign_directory.glob("sample_*.npy")
        )

        print(
            f"Loading {len(sample_files)} "
            f"samples for: {sign_name}"
        )

        for sample_file in sample_files:
            sample = np.load(sample_file)

            if sample.shape != EXPECTED_SHAPE:
                print(
                    f"Skipping {sample_file}: "
                    f"invalid shape {sample.shape}"
                )
                continue

            # Convert 30 × 132 into one row of 3,960 values
            flattened_sample = sample.flatten()

            features.append(flattened_sample)
            labels.append(label_index)

    return (
        np.array(features, dtype=np.float32),
        np.array(labels, dtype=np.int64)
    )


def main():
    features, labels = load_dataset()

    print("\nDataset information")
    print("-" * 40)
    print(f"Total samples: {len(features)}")
    print(f"Input shape: {features.shape}")
    print(f"Label shape: {labels.shape}")

    if len(features) == 0:
        print("Error: No valid training samples found.")
        return

    x_train, x_test, y_train, y_test = (
        train_test_split(
            features,
            labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels
        )
    )

    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1
    )

    print("\nTraining model...")
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nModel results")
    print("-" * 40)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=SIGNS,
            zero_division=0
        )
    )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    model_package = {
        "model": model,
        "signs": SIGNS,
        "sequence_length": EXPECTED_SHAPE[0],
        "feature_count": EXPECTED_SHAPE[1]
    }

    joblib.dump(
        model_package,
        MODEL_PATH
    )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
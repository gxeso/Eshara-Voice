import time

import joblib
import numpy as np


class SignRecognizer:

    def __init__(
        self,
        model_path="models/eshara_rf.joblib",
        recording_duration=2.0
    ):
        model_package = joblib.load(model_path)

        self.model = model_package["model"]
        self.signs = model_package["signs"]

        self.sequence_length = (
            model_package["sequence_length"]
        )

        self.feature_count = (
            model_package["feature_count"]
        )

        self.recording_duration = recording_duration

        self.current_sequence = []
        self.capture_start_time = None
        self.is_capturing = False

    def start_capture(self):
        self.current_sequence = []
        self.capture_start_time = time.monotonic()
        self.is_capturing = True

    def add_frame(self, feature_vector):
        if not self.is_capturing:
            return None

        self.current_sequence.append(
            feature_vector
        )

        elapsed_time = (
            time.monotonic()
            - self.capture_start_time
        )

        if elapsed_time < self.recording_duration:
            return None

        prediction = self.predict_sequence()

        self.is_capturing = False
        self.capture_start_time = None
        self.current_sequence = []

        return prediction

    def predict_sequence(self):
        sequence = self.resample_sequence(
            self.current_sequence
        )

        # No hand was detected during the entire capture
        if sequence is None:
            return {
                "sign": "unknown",
                "confidence": 0.0
            }

        flattened_sequence = (
            sequence.flatten().reshape(1, -1)
        )

        predicted_index = int(
            self.model.predict(
                flattened_sequence
            )[0]
        )

        probabilities = self.model.predict_proba(
            flattened_sequence
        )[0]

        confidence = float(
            probabilities[predicted_index]
        )

        return {
            "sign": self.signs[predicted_index],
            "confidence": confidence
        }

    def resample_sequence(self, sequence):
        sequence_array = np.array(
            sequence,
            dtype=np.float32
        )

        if (
            sequence_array.ndim != 2
            or len(sequence_array) == 0
        ):
            return None

        # A completely zero row means neither hand
        # was detected during that frame.
        empty_frames = np.all(
            np.isclose(sequence_array, 0.0),
            axis=1
        )

        valid_frames = ~empty_frames

        # Reject a sequence where no hands were ever detected
        if not np.any(valid_frames):
            return None

        # Repair temporary detection failures
        if np.any(empty_frames):
            frame_positions = np.arange(
                len(sequence_array)
            )

            valid_positions = frame_positions[
                valid_frames
            ]

            for feature_index in range(
                sequence_array.shape[1]
            ):
                valid_values = sequence_array[
                    valid_frames,
                    feature_index
                ]

                sequence_array[
                    :,
                    feature_index
                ] = np.interp(
                    frame_positions,
                    valid_positions,
                    valid_values
                )

        original_frame_count = len(
            sequence_array
        )

        original_positions = np.linspace(
            0,
            1,
            original_frame_count
        )

        new_positions = np.linspace(
            0,
            1,
            self.sequence_length
        )

        resampled_features = []

        for feature_index in range(
            self.feature_count
        ):
            feature_values = sequence_array[
                :,
                feature_index
            ]

            resampled_values = np.interp(
                new_positions,
                original_positions,
                feature_values
            )

            resampled_features.append(
                resampled_values
            )

        return np.array(
            resampled_features,
            dtype=np.float32
        ).T

    def get_progress(self):
        if not self.is_capturing:
            return 0.0

        elapsed_time = (
            time.monotonic()
            - self.capture_start_time
        )

        return min(
            elapsed_time / self.recording_duration,
            1.0
        )
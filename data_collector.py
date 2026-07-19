import time
from pathlib import Path

import numpy as np


class DataCollector:

    def __init__(
        self,
        sign_name,
        sequence_length=30,
        recording_duration=2.0,
        data_directory="data"
    ):
        self.sign_name = sign_name
        self.sequence_length = sequence_length
        self.recording_duration = recording_duration

        self.sign_directory = (
            Path(data_directory) / sign_name
        )

        self.sign_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.current_sequence = []
        self.is_recording = False
        self.recording_start_time = None

    def start_recording(self):
        self.current_sequence = []
        self.is_recording = True
        self.recording_start_time = time.monotonic()

        print(f"Recording sign: {self.sign_name}")

    def add_frame(self, feature_vector):
        if not self.is_recording:
            return False

        self.current_sequence.append(feature_vector)

        elapsed_time = (
            time.monotonic() - self.recording_start_time
        )

        if elapsed_time >= self.recording_duration:
            self.save_sequence()
            self.is_recording = False
            self.recording_start_time = None

            return True

        return False

    def resample_sequence(self):
        sequence_array = np.array(
            self.current_sequence,
            dtype=np.float32
        )

        original_frame_count = len(sequence_array)

        if original_frame_count == 1:
            return np.repeat(
                sequence_array,
                self.sequence_length,
                axis=0
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

        for feature_index in range(sequence_array.shape[1]):
            feature_values = sequence_array[:, feature_index]

            resampled_values = np.interp(
                new_positions,
                original_positions,
                feature_values
            )

            resampled_features.append(resampled_values)

        return np.array(
            resampled_features,
            dtype=np.float32
        ).T

    def save_sequence(self):
        sample_number = self.get_next_sample_number()

        file_path = self.sign_directory / (
            f"sample_{sample_number:03d}.npy"
        )

        sequence_array = self.resample_sequence()

        np.save(file_path, sequence_array)

        print(f"Saved: {file_path}")
        print(
            f"Original frames: "
            f"{len(self.current_sequence)}"
        )
        print(f"Saved shape: {sequence_array.shape}")

    def get_recording_progress(self):
        if not self.is_recording:
            return 0.0

        elapsed_time = (
            time.monotonic() - self.recording_start_time
        )

        return min(
            elapsed_time / self.recording_duration,
            1.0
        )

    def get_next_sample_number(self):
        existing_files = list(
            self.sign_directory.glob("sample_*.npy")
        )

        if not existing_files:
            return 1

        existing_numbers = []

        for file_path in existing_files:
            number_text = file_path.stem.replace(
                "sample_",
                ""
            )

            if number_text.isdigit():
                existing_numbers.append(int(number_text))

        if not existing_numbers:
            return 1

        return max(existing_numbers) + 1

    def get_sample_count(self):
        existing_files = list(
            self.sign_directory.glob("sample_*.npy")
        )

        return len(existing_files)
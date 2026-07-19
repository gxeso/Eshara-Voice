from pathlib import Path

import numpy as np

SIGN_NAME = "swelling"

EXPECTED_SHAPE = (30, 132)

sign_directory = Path("data") / SIGN_NAME
sample_files = sorted(sign_directory.glob("sample_*.npy"))

if not sample_files:
    print(f"No samples found inside: {sign_directory}")
    exit()

invalid_samples = []
samples_with_missing_hands = []
movement_scores = []

print(f"Validating {len(sample_files)} samples for: {SIGN_NAME}")
print("-" * 50)

for sample_file in sample_files:
    sample = np.load(sample_file)

    if sample.shape != EXPECTED_SHAPE:
        invalid_samples.append(
            (sample_file.name, sample.shape)
        )
        continue

    # A completely zero frame means no hands were detected
    zero_frames = np.sum(
        np.all(np.isclose(sample, 0.0), axis=1)
    )

    if zero_frames > 0:
        samples_with_missing_hands.append(
            (sample_file.name, zero_frames)
        )

    movement = np.mean(
        np.abs(np.diff(sample, axis=0))
    )

    movement_scores.append(
        (sample_file.name, movement)
    )

print(f"Total samples: {len(sample_files)}")
print(f"Invalid shapes: {len(invalid_samples)}")
print(
    "Samples containing completely empty frames:",
    len(samples_with_missing_hands)
)

if movement_scores:
    movement_values = [
        score for _, score in movement_scores
    ]

    print(f"Minimum movement: {min(movement_values):.8f}")
    print(f"Average movement: {np.mean(movement_values):.8f}")
    print(f"Maximum movement: {max(movement_values):.8f}")

if invalid_samples:
    print("\nInvalid samples:")

    for file_name, shape in invalid_samples:
        print(f"  {file_name}: {shape}")

if samples_with_missing_hands:
    print("\nSamples with empty frames:")

    for file_name, zero_frames in samples_with_missing_hands:
        print(
            f"  {file_name}: "
            f"{zero_frames} empty frame(s)"
        )

print("\nFive samples with the lowest movement:")

for file_name, movement in sorted(
    movement_scores,
    key=lambda item: item[1]
)[:5]:
    print(f"  {file_name}: {movement:.8f}")
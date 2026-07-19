import shutil
from pathlib import Path

import numpy as np


SIGN_NAME = "hospital_ward"

DATA_DIRECTORY = Path("data") / SIGN_NAME
BACKUP_DIRECTORY = Path("data_backup") / SIGN_NAME

BACKUP_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

sample_files = sorted(
    DATA_DIRECTORY.glob("sample_*.npy")
)

repaired_count = 0


for sample_file in sample_files:
    sequence = np.load(sample_file)

    empty_frames = np.all(
        np.isclose(sequence, 0.0),
        axis=1
    )

    empty_count = int(np.sum(empty_frames))

    if empty_count == 0:
        continue

    valid_frames = ~empty_frames

    if not np.any(valid_frames):
        print(
            f"Cannot repair {sample_file.name}: "
            "every frame is empty"
        )
        continue

    # Preserve the original before changing it
    backup_path = BACKUP_DIRECTORY / sample_file.name

    if not backup_path.exists():
        shutil.copy2(
            sample_file,
            backup_path
        )

    frame_positions = np.arange(
        len(sequence)
    )

    valid_positions = frame_positions[
        valid_frames
    ]

    repaired_sequence = sequence.copy()

    for feature_index in range(sequence.shape[1]):
        valid_values = sequence[
            valid_frames,
            feature_index
        ]

        repaired_sequence[
            :,
            feature_index
        ] = np.interp(
            frame_positions,
            valid_positions,
            valid_values
        )

    np.save(
        sample_file,
        repaired_sequence.astype(np.float32)
    )

    repaired_count += 1

    print(
        f"Repaired {sample_file.name}: "
        f"{empty_count} empty frame(s)"
    )


print("-" * 50)
print(f"Repaired samples: {repaired_count}")
print(f"Original backups: {BACKUP_DIRECTORY}")
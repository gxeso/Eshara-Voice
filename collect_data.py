import time

import cv2

from camera_manager import CameraManager
from data_collector import DataCollector
from hand_detector import HandDetector

AVAILABLE_SIGNS = [
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

SEQUENCE_LENGTH = 30
RECORDING_DURATION = 2.0
TARGET_SAMPLES = 40


print("Available Emirati signs:")

for index, sign in enumerate(AVAILABLE_SIGNS, start=1):
    print(f"{index}. {sign}")

choice = input("Choose a sign number: ").strip()

if not choice.isdigit():
    print("Error: Please enter a number.")
    exit()

choice_index = int(choice) - 1

if choice_index < 0 or choice_index >= len(AVAILABLE_SIGNS):
    print("Error: Invalid sign number.")
    exit()

selected_sign = AVAILABLE_SIGNS[choice_index]

collector = DataCollector(
    sign_name=selected_sign,
    sequence_length=SEQUENCE_LENGTH,
    recording_duration=RECORDING_DURATION
)

camera_manager = CameraManager()
hand_detector = HandDetector()

countdown_active = False
countdown_start = 0

if camera_manager.open_camera():

    while True:
        frame = camera_manager.get_frame()

        if frame is None:
            break

        frame, results = hand_detector.detect_hands(frame)

        feature_vector = (
            hand_detector.get_feature_vector(results)
        )

        sample_count = collector.get_sample_count()

        cv2.putText(
            frame,
            f"Sign: {selected_sign}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {sample_count}/{TARGET_SAMPLES}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        if countdown_active:
            elapsed_time = time.monotonic() - countdown_start
            remaining_time = max(
                1,
                3 - int(elapsed_time)
            )

            if elapsed_time >= 3:
                countdown_active = False
                collector.start_recording()
            else:
                cv2.putText(
                    frame,
                    f"Starting in: {remaining_time}",
                    (180, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    3
                )

        if collector.is_recording:
            collector.add_frame(feature_vector)

            progress = int(
                collector.get_recording_progress() * 100
            )

            cv2.putText(
                frame,
                f"RECORDING: {progress}%",
                (160, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

        elif not countdown_active:
            cv2.putText(
                frame,
                "Press R to record",
                (180, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

        cv2.imshow(
            "Eshara Voice - Data Collection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            if (
                not collector.is_recording
                and not countdown_active
            ):
                countdown_active = True
                countdown_start = time.monotonic()

        elif key == ord("q"):
            break

        if collector.get_sample_count() >= TARGET_SAMPLES:
            print(
                f"Completed {TARGET_SAMPLES} samples "
                f"for {selected_sign}."
            )
            break

    hand_detector.close()
    camera_manager.release_camera()
import time

import cv2

from camera_manager import CameraManager
from hand_detector import HandDetector
from sign_recognizer import SignRecognizer
from speech_controller import SpeechController


def format_sign_name(sign_name):
    """
    Convert an internal label into a readable name.

    Example:
    hospital_ward -> Hospital Ward
    """
    return sign_name.replace("_", " ").title()


camera_manager = CameraManager()
hand_detector = HandDetector()
sign_recognizer = SignRecognizer()
speech_controller = SpeechController()

countdown_active = False
countdown_start = None

last_prediction = "None"
last_confidence = 0.0

CONFIDENCE_THRESHOLD = 0.30


if camera_manager.open_camera():

    while True:
        frame = camera_manager.get_frame()

        if frame is None:
            break

        # Detect hands and draw landmarks
        frame, results = hand_detector.detect_hands(
            frame
        )

        # Convert detected hands into 132 features
        feature_vector = (
            hand_detector.get_feature_vector(results)
        )

        # Display countdown before capture
        if countdown_active:
            elapsed_time = (
                time.monotonic() - countdown_start
            )

            remaining_time = max(
                1,
                3 - int(elapsed_time)
            )

            if elapsed_time >= 3:
                countdown_active = False
                sign_recognizer.start_capture()

            else:
                cv2.putText(
                    frame,
                    f"Starting in: {remaining_time}",
                    (160, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    3
                )

        # Capture a two-second sign sequence
        if sign_recognizer.is_capturing:
            prediction = sign_recognizer.add_frame(
                feature_vector
            )

            progress = int(
                sign_recognizer.get_progress() * 100
            )

            cv2.putText(
                frame,
                f"PERFORM SIGN: {progress}%",
                (120, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

            # Process completed prediction
            if prediction is not None:
                last_prediction = prediction["sign"]

                last_confidence = prediction[
                    "confidence"
                ]

                if (
                    last_confidence
                    >= CONFIDENCE_THRESHOLD
                ):
                    speech_controller.speak_sign(
                        last_prediction
                    )

        # Show instructions while idle
        elif not countdown_active:
            cv2.putText(
                frame,
                "Press R to recognize",
                (130, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

        # Convert internal label into readable text
        display_prediction = format_sign_name(
            last_prediction
        )

        # Display latest prediction
        cv2.putText(
            frame,
            f"Prediction: {display_prediction}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )

        # Display prediction confidence
        cv2.putText(
            frame,
            f"Confidence: {last_confidence:.2%}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # Warn when confidence is too low
        if (
            last_prediction != "None"
            and last_confidence
            < CONFIDENCE_THRESHOLD
        ):
            cv2.putText(
                frame,
                "Low confidence - please try again",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Eshara Voice - Recognition",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Press R to start recognition
        if key == ord("r"):
            if (
                not countdown_active
                and not sign_recognizer.is_capturing
            ):
                countdown_active = True

                countdown_start = (
                    time.monotonic()
                )

        # Press Q to close
        elif key == ord("q"):
            break

    speech_controller.close()
    hand_detector.close()
    camera_manager.release_camera()
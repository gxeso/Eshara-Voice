import cv2

from camera_manager import CameraManager
from hand_detector import HandDetector


camera_manager = CameraManager()
hand_detector = HandDetector()

if camera_manager.open_camera():

    while True:
        frame = camera_manager.get_frame()

        if frame is None:
            break

        # Detect and draw hand landmarks
        frame, results = hand_detector.detect_hands(frame)

        # Convert the detected hands into 126 numeric features
        feature_vector = hand_detector.get_feature_vector(results)

        # Display the number of features on the camera window
        cv2.putText(
            frame,
            f"Features: {len(feature_vector)}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.imshow("Eshara Voice", frame)

        # Press Q to close the application
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hand_detector.close()
    camera_manager.release_camera()
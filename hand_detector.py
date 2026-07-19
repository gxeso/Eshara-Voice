import time

import cv2
import mediapipe as mp


class HandDetector:

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),          # Index
        (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
        (13, 17), (17, 18), (18, 19), (19, 20),  # Little
        (0, 17)
    ]

    def __init__(
        self,
        model_path="hand_landmarker.task",
        max_hands=2,
        detection_confidence=0.7,
        tracking_confidence=0.7
    ):
        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        self.detector = (
            mp.tasks.vision.HandLandmarker.create_from_options(options)
        )

        self.start_time = time.monotonic()
        self.last_timestamp_ms = -1

    def detect_hands(self, frame):
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mediapipe_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        current_timestamp_ms = int(
            (time.monotonic() - self.start_time) * 1000
        )

        # MediaPipe requires timestamps to increase
        # for every processed video frame.
        timestamp_ms = max(
            current_timestamp_ms,
            self.last_timestamp_ms + 1
        )

        self.last_timestamp_ms = timestamp_ms

        result = self.detector.detect_for_video(
            mediapipe_image,
            timestamp_ms
        )

        self.draw_landmarks(frame, result)

        return frame, result

    def draw_landmarks(self, frame, result):
        frame_height, frame_width, _ = frame.shape

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):
            points = []

            for landmark in hand_landmarks:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                points.append((x, y))

            # Draw connections between landmarks
            for start_index, end_index in self.HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start_index],
                    points[end_index],
                    (0, 255, 0),
                    2
                )

            # Draw and number the 21 landmarks
            for landmark_index, point in enumerate(points):
                cv2.circle(
                    frame,
                    point,
                    5,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    str(landmark_index),
                    (point[0] + 5, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

            handedness = result.handedness[hand_index][0]

            detected_hand = handedness.category_name
            confidence = handedness.score

            # Correct handedness because the frame is mirrored
            if detected_hand == "Left":
                hand_name = "Right"
            else:
                hand_name = "Left"

            wrist_x, wrist_y = points[0]

            cv2.putText(
                frame,
                f"{hand_name}: {confidence:.2f}",
                (wrist_x, wrist_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )

        cv2.putText(
            frame,
            f"Hands detected: {len(result.hand_landmarks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    def get_landmark_data(self, result):
        hands_data = []

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):
            detected_hand = (
                result.handedness[hand_index][0].category_name
            )

            if detected_hand == "Left":
                hand_name = "Right"
            else:
                hand_name = "Left"

            landmarks = []

            for landmark in hand_landmarks:
                landmarks.append([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            hands_data.append({
                "hand": hand_name,
                "landmarks": landmarks
            })

        return hands_data

    def get_feature_vector(self, result):
        # Each hand contains:
        # 3 absolute wrist coordinates
        # 21 landmarks × 3 relative coordinates
        # Total: 66 values per hand
        left_hand = [0.0] * 66
        right_hand = [0.0] * 66

        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):
            detected_hand = (
                result.handedness[hand_index][0].category_name
            )

            # Correct handedness because the frame is mirrored
            if detected_hand == "Left":
                hand_name = "Right"
            else:
                hand_name = "Left"

            wrist = hand_landmarks[0]

            # Preserve the hand's position in the image
            hand_features = [
                wrist.x,
                wrist.y,
                wrist.z
            ]

            # Preserve the shape of the hand relative to its wrist
            for landmark in hand_landmarks:
                hand_features.extend([
                    landmark.x - wrist.x,
                    landmark.y - wrist.y,
                    landmark.z - wrist.z
                ])

            if hand_name == "Left":
                left_hand = hand_features
            else:
                right_hand = hand_features

        # 66 left-hand values + 66 right-hand values
        return left_hand + right_hand

    def close(self):
        self.detector.close()
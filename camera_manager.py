import cv2


class CameraManager:

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.camera = None

    def open_camera(self):
        self.camera = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW
        )

        if not self.camera.isOpened():
            print("Error: Could not open the camera.")
            return False

        print("Camera opened successfully.")
        return True

    def get_frame(self):
        success, frame = self.camera.read()

        if not success:
            print("Error: Could not capture a frame.")
            return None

        frame = cv2.flip(frame, 1)

        return frame

    def release_camera(self):
        if self.camera is not None:
            self.camera.release()

        cv2.destroyAllWindows()
        print("Camera closed.")
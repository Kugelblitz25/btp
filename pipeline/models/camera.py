import cv2
import numpy as np
import time
from typing import Generator, Optional


class Camera:
    def __init__(self, url: str, retry_interval: int = 5):
        self.url = url
        self.retry_interval = retry_interval
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False

    def connect(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.url)
            if not self.cap.isOpened():
                return False
            self.is_running = True
            return True
        except Exception as e:
            print(f"Error connecting to camera: {e}")
            return False

    def disconnect(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.is_running = False
        self.cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.is_running or self.cap is None:
            if not self.connect():
                return None

        ret, frame = self.cap.read()
        if not ret:
            self.disconnect()
            return None

        return frame

    def frames(self) -> Generator[np.ndarray, None, None]:
        while True:
            if not (self.is_running or self.connect()):
                print(
                    f"Connection failed. Retrying in {self.retry_interval} seconds..."
                )
                time.sleep(self.retry_interval)
                continue

            frame = self.get_frame()
            if frame is None:
                print("Video feed ended.")
                break

            yield frame

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

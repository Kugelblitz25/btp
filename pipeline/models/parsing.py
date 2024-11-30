import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
import webcolors

Color = tuple[int, int, int]


class HumanProcessor:
    def __init__(self):
        """Initialize MediaPipe pose detector and other required components"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        )

    def get_color_name(self, color: Color) -> str:
        """Convert RGB color to nearest color name"""
        try:
            hex_value = webcolors.rgb_to_hex(color)
            return webcolors.hex_to_name(hex_value)
        except ValueError:
            return min(
                (
                    (webcolors.name_to_rgb(name), name)
                    for name in webcolors.names("css2")
                ),
                key=lambda x: sum((a - b) ** 2 for a, b in zip(x[0], color)),
            )[1]

    def get_mode_color(self, corners: np.ndarray, frame: np.ndarray) -> dict:
        mask = np.zeros_like(frame, dtype="uint8")
        cv2.fillPoly(mask, [corners.reshape((-1, 1, 2))], 255)
        xcords, ycoords = np.where(mask == 255)[:2]
        pixels = frame[xcords, ycoords, :]
        if len(pixels) == 0:
            return {"name": "unknown", "rgb": (0, 0, 0)}

        pixels = pixels.reshape(-1, 3)
        mode_color = Counter(map(tuple, pixels)).most_common(1)[0][0]
        rgb_color = (mode_color[2], mode_color[1], mode_color[0])  # BGR to RGB
        color_name = self.get_color_name(rgb_color)

        return {"name": color_name, "rgb": rgb_color}

    def get_keypoints(self, landmarks: list, w: float, h: float) -> dict:
        lm_points = {
            "right_hip": (
                int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                int(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * h),
            ),
            "left_hip": (
                int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                int(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h),
            ),
            "right_shoulder": (
                int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h),
            ),
            "left_shoulder": (
                int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h),
            ),
            "right_knee": (
                int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w),
                int(landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h),
            ),
            "left_knee": (
                int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w),
                int(landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h),
            ),
            "right_ankle": (
                int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w),
                int(landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h),
            ),
            "left_ankle": (
                int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w),
                int(landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h),
            ),
        }
        return lm_points

    def process_frame(
        self, frame: np.ndarray, detections: list[dict]
    ) -> tuple[np.ndarray, dict]:
        results_dict = {}
        output_frame = frame.copy()
        H, W, _ = frame.shape

        for det in detections:
            person_id = det["id"]
            bbox = det["bbox"]
            conf = det["conf"]

            x1, y1, x2, y2 = map(int, bbox)
            person_frame = frame[y1:y2, x1:x2]

            # Process with MediaPipe
            try:
                rgb_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
            except cv2.error:
                continue

            if not results.pose_landmarks:
                continue

            # Get frame dimensions and create mask
            h, w, _ = person_frame.shape

            # Extract landmarks
            lm_points = self.get_keypoints(results.pose_landmarks.landmark, w, h)

            # Process torso
            torso_pts = np.array(
                [
                    lm_points["right_hip"],
                    lm_points["right_shoulder"],
                    lm_points["left_shoulder"],
                    lm_points["left_hip"],
                ],
                dtype=np.int32,
            )
            torso_color = self.get_mode_color(torso_pts, person_frame)

            # Process lower body
            lower_pts = np.array(
                [
                    lm_points["right_hip"],
                    lm_points["left_hip"],
                    lm_points["left_knee"],
                    lm_points["right_knee"],
                ],
                dtype=np.int32,
            )
            lower_color = self.get_mode_color(lower_pts, person_frame)

            # Extract face
            face_bbox = [
                x1,
                y1,
                x1 + min(lm_points["left_shoulder"][1], lm_points["right_shoulder"][1]),
                y1 + h,
            ]

            left_foot = [
                lm_points["left_ankle"][0] + x1,
                lm_points["left_ankle"][1] + y1,
            ]

            right_foot = [
                lm_points["right_ankle"][0] + x1,
                lm_points["right_ankle"][1] + y1,
            ]

            # Draw on output frame
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output_frame,
                f"ID: {person_id} ({conf:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Store results
            results_dict[person_id] = {
                "torso_color": torso_color,
                "lower_body_color": lower_color,
                "foot_coordinates": {
                    "left": left_foot,
                    "right": right_foot,
                },
                "face_bbox": face_bbox,
                "confidence": conf,
            }

        return output_frame, results_dict

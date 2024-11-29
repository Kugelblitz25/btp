import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class PersonTracker:
    def __init__(self, args):
        self.args = args
        self.person_database = {}
        self.similarity_dict = defaultdict(int)

    def load_yolo_model(self):
        model_dir = Path(__file__).parent / "models"
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / self.args["model"]
        return YOLO(model_path, verbose=False)

    def load_deepsort_model(self):
        return DeepSort(max_age=self.args["max_age"], nn_budget=self.args["nn_budget"])

    def process_detections(self, results: list) -> list:
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf > self.args["confidence_threshold"]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append([[x1, y1, x2 - x1, y2 - y1], conf])
        return detections

    def process_tracks(self, tracks: list, confs: list[float]) -> list[dict]:
        persons = []
        for track, conf in zip(tracks, confs):
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)

            current_feature = np.array(track.features).reshape(1, -1)

            reidentified = False
            for existing_id, existing_feature in self.person_database.items():
                similarity = cosine_similarity(existing_feature, current_feature)[0][0]
                if similarity > self.args["similarity_threshold"]:
                    self.similarity_dict[(track_id, existing_id)] += 1
                    if self.similarity_dict[(track_id, existing_id)] > 10:
                        reidentified = True
                        track_id = existing_id
                        break

            if not reidentified:
                self.person_database[track_id] = current_feature

            persons.append({"id": track_id, "bbox": [x1, y1, x2, y2], "conf": conf})
        return persons

    def init(self):
        self.yolo = self.load_yolo_model()
        self.deepsort = self.load_deepsort_model()

    def track(self, frame: np.ndarray):
        detections = self.yolo(frame)
        detections = self.process_detections(detections)
        confs = [detection[1] for detection in detections]
        tracks = self.deepsort.update_tracks(detections, frame=frame)
        tracks = self.process_tracks(tracks, confs)
        return tracks

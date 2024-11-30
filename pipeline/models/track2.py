import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch


class PersonTracker:
    def __init__(self, args):
        """
        Initialize PersonTracker with configuration parameters
        """
        self.args = args
        self.yolo = self.load_yolo_model()

    def load_yolo_model(self):
        """
        Load YOLO model from specified path
        """
        model_dir = Path(__file__).parent / "models"
        model_dir.mkdir(exist_ok=True, parents=True)
        model_path = model_dir / self.args["model"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("Using CPU")

        model = YOLO(model_path, verbose=False)
        return model.to(device)

    def process_tracks(self, results) -> list[dict]:
        """
        Process YOLO tracking results
        """
        persons = []

        if not results or not results[0].boxes:
            return persons

        boxes = results[0].boxes
        if boxes.id is None:  # No tracks found
            return persons

        # Get boxes and track IDs
        track_boxes = boxes.xywh.cpu().numpy()  # x, y, w, h format
        track_ids = boxes.id.int().cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(track_boxes, track_ids, confidences):
            if conf < self.args["confidence_threshold"]:
                continue

            x, y, w, h = box
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            persons.append(
                {
                    "id": int(track_id),
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(conf),
                }
            )

        return persons

    def track(self, frame: np.ndarray):
        """
        Process a single frame and return tracking results
        """
        results = self.yolo.track(
            frame,
            persist=True,
            classes=[0],  # Only track persons
            verbose=False,
            tracker=Path(__file__).parent / "models" / self.args["tracker"],
        )

        return self.process_tracks(results)

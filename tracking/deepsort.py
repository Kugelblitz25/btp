import argparse
import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


class PersonTracker:
    def __init__(self, args):
        self.args = args
        self.person_database = {}

    def load_model(self):
        from ultralytics import YOLO
        model_path = Path("models") / self.args.model
        return YOLO(model_path, verbose=False)

    def initialize_tracker(self):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        return DeepSort(max_age=self.args.max_age, nn_budget=self.args.nn_budget)

    def process_detections(self, results):
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf > self.args.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append([[x1, y1, x2 - x1, y2 - y1], conf])
        return detections

    def estimate_face_from_keypoints(self, keypoints, bbox):
        if keypoints.shape[0] < 7:
            return None, None
        
        left_shoulder, right_shoulder = keypoints[5], keypoints[6]

        if left_shoulder[2] > self.args.confidence_threshold and right_shoulder[2] > self.args.confidence_threshold:
            s1x, s1y = left_shoulder[:2]
            s2x, s2y = right_shoulder[:2]
            y2 = int(max(s1y, s2y))
            x1, x2 = sorted([int(s1x), int(s2x)])
            x1 = max(int(x1 - 0.5*(x2-x1)), bbox[0])
            x2 = min(int(x2 + 0.5*(x2-x1)), bbox[2])

            return (x1, bbox[1]), (x2, y2)
        
        return None, None

    def draw_tracks(self, frame, tracks):
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)

            current_feature = np.array(track.features).reshape(1, -1)

            reidentified = False
            for existing_id, existing_feature in self.person_database.items():
                similarity = cosine_similarity(existing_feature, current_feature)[0][0]
                if similarity > self.args.similarity_threshold and existing_id != track_id:
                    reidentified = True
                    print(f"Reidentified Track {track_id} as existing Track {existing_id}")
                    track_id = existing_id
                    break

            if not reidentified:
                self.person_database[track_id] = current_feature

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame

    def draw_faces(self, frame, results):
        for result in results:      
            keypoints = result.keypoints.data[0].cpu().numpy()
            bbox = result.boxes.xyxy[0].cpu().numpy().astype(int)
            p1, p2 = self.estimate_face_from_keypoints(keypoints, bbox)
            
            if p1 is not None:
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)

        return frame

    def run(self):
        self.model = self.load_model()
        self.tracker = self.initialize_tracker()
        cap = cv2.VideoCapture(self.args.video)
        if self.args.output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.output_writer = cv2.VideoWriter(self.args.output, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model(frame)
            detections = self.process_detections(results)
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            frame = self.draw_tracks(frame, tracks)
            frame = self.draw_faces(frame, results)
            
            cv2.imshow('Frame', frame)
            if self.output_writer:
                self.output_writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if self.output_writer:
            self.output_writer.release()
        cv2.destroyAllWindows()
        print("Final tracked IDs:", self.person_database.keys())

def parse_arguments():
    parser = argparse.ArgumentParser(description="Person Tracker with YOLO and DeepSort")
    parser.add_argument("--model", type=str, default="yolov8s-pose.pt", help="Path to the YOLO model")
    parser.add_argument("--video", type=str, default="videos/ncair.mp4", help="Path to the input video")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output video (optional)")
    parser.add_argument("--confidence_threshold", type=float, default=0.6, help="Confidence threshold for detections")
    parser.add_argument("--max_age", type=int, default=30, help="Maximum age for tracks in DeepSort")
    parser.add_argument("--nn_budget", type=int, default=100, help="NN budget for DeepSort")
    parser.add_argument("--similarity_threshold", type=float, default=0.7, help="Similarity threshold for re-identification")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    tracker = PersonTracker(args)
    tracker.run()
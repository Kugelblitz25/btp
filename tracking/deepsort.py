import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "D:/Downloads/op_vd_4.mp4" 
CONFIDENCE_THRESHOLD = 0.5
MAX_AGE = 30
NN_BUDGET = 100

def load_model():
    return YOLO(MODEL_PATH)

def initialize_tracker():
    return DeepSort(max_age=MAX_AGE, nn_budget=NN_BUDGET)

def process_detections(results, confidence_threshold):
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls, conf = int(box.cls[0]), float(box.conf[0])
            if cls == 0 and conf > confidence_threshold:  # Class 0 is 'person' in COCO dataset
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append([[x1, y1, x2-x1, y2-y1], conf])
    return detections

def draw_tracks(frame, tracks):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame



def main():
    model = load_model()
    tracker = initialize_tracker()
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = process_detections(results, CONFIDENCE_THRESHOLD)
        tracks = tracker.update_tracks(detections, frame=frame)

        frame = draw_tracks(frame, tracks)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
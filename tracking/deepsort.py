import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

MODEL_PATH = "yolov8n.pt"  # Make sure to download this model
FACE_MODEL_PATH = "yolov8n-pose.pt"
VIDEO_PATH = "videos/ncair.mp4"
CONFIDENCE_THRESHOLD = 0.3
MAX_AGE = 30
NN_BUDGET = 100

face_model = YOLO(FACE_MODEL_PATH)

def load_model():
    return YOLO(MODEL_PATH)

def initialize_tracker():
    return DeepSort(max_age=MAX_AGE, nn_budget=NN_BUDGET)

def process_detections(results, confidence_threshold):
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf > confidence_threshold:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                detections.append([[x1, y1, x2-x1, y2-y1], conf])
    return detections

def estimate_face_from_keypoints(img):
    # Get left and right shoulder keypoints
    results = face_model(img)
    keypoints = results[0].keypoints.data[0].cpu().numpy()

    if keypoints.shape[0] < 7:
        return None, None, None
    
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    if left_shoulder[2] > CONFIDENCE_THRESHOLD and right_shoulder[2] > CONFIDENCE_THRESHOLD:
        s1x, s1y = left_shoulder[:2]
        s2x, s2y = right_shoulder[:2]
        max_y = max(s1y, s2y)
        x1 = min(s1x, s2x)
        x2 = max(s1x, s2x)
        x1 = int(x1 - 0.5*(x2-x1))
        x2 = int(x2 + 0.5*(x2-x1))

        # Estimate face bounding box    
        return int(max_y), x1, x2
    
    return None, None, None

def draw_tracks_and_faces(frame, tracks):
    for track in zip(tracks):
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw person bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Estimate and draw face bounding box
        max_y, fx1, fx2 = estimate_face_from_keypoints(frame[int(y1):int(y2), int(x1):int(x2)])
        if max_y is not None:
            fx, fy, fw, fh = fx1+x1, max_y, fx2-fx1, max_y-y1
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)
            cv2.putText(frame, 'Face', (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    
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
        
        frame = draw_tracks_and_faces(frame, tracks)
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
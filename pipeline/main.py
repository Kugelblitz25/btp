from models.camera import Camera
from models.parsing import HumanProcessor
from models.track2 import PersonTracker
import json
import cv2

with open("pipeline/tracker_config.json") as f:
    config = json.load(f)

tracker = PersonTracker(config)
parser = HumanProcessor()
vid_loc = "rtsp://ncair:ncair@10.185.153.173/cam/realmonitor?channel=2&subtype=1"
vid_loc = "test_videos/test_vid_from_ncair_1.mp4"
with Camera(vid_loc) as camera:
    for frame in camera.frames():
        results = tracker.track(frame)
        frame, results = parser.process_frame(frame, results)
        print(results)
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()

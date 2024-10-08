from ultralytics import YOLO
import cv2
from time import perf_counter

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n-pose.pt')  # 'n' for nano, can be 's', 'm', 'l', 'x' for other sizes

# Load an image
img = cv2.imread('../images/yolo_test1.png')

# Perform detection
start = perf_counter()
results = model(img)[0]
end = perf_counter()
print(f"Detection took {end - start} seconds")
class_names = results.names

boxes = results.boxes
bounds = boxes.xyxy.numpy()
classes = boxes.cls.numpy()
keypoints = results.keypoints.xy.numpy()

for (x1, y1, x2, y2), cls,kpts in zip(bounds, classes, keypoints):
    sub_img = img[int(y1):int(y2), int(x1):int(x2)]
    for point in kpts:
        x, y = point[0] - x1, point[1] - y1
        cv2.circle(sub_img, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.imwrite(f'{class_names[cls]}/{x2}.png', sub_img)
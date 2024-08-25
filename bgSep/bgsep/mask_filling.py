import cv2
from algoTester import VideoWriter
import numpy as np

vid = 'videos/seg_mask.mp4'
# writer = VideoWriter('videos/seg_mask2.mp4', [320, 480, 30])

cap  = cv2.VideoCapture(vid)
kernel_ex =cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  
kernel_er =cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.medianBlur(frame, 3)
    dilated = cv2.dilate(frame, kernel_ex, iterations=1)
    closing = cv2.erode(dilated, kernel_er, iterations=1)
    cv2.imshow('Frame', closing)
    # writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# writer.close()
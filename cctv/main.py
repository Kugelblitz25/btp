import cv2

cap = cv2.VideoCapture(
    "rtsp://ncair:ncair@10.185.153.173/cam/realmonitor?channel=3&subtype=1"
)

while True:
    ret, frame = cap.read()
    if ret != 0:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

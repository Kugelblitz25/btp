import cv2
import sys
import select
import tty
import termios

# Setup for non-blocking keypress
fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
tty.setcbreak(fd)

cap = cv2.VideoCapture(
    "rtsp://ncair:ncair@10.185.153.23/cam/realmonitor?channel=2&subtype=0"
)

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}, Width: {width}, height: {height}")
print("Press 'q' to quit...")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('calibration/z0/output.avi', fourcc, fps, (width, height))

try:
    while True:
        ret, frame = cap.read()
        if ret:
            out.write(frame)

            # Non-blocking check for keypress
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == 'q':
                    break
        else:
            break
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    cap.release()
    out.release()

import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "D:/Downloads/op_vd_4.mp4"                                               # Video Path
threshold = 100                                                                 # Threshold
max_frame_diffs=[]                                                              # Stores maximum of pixel differences between consecutive frames
frame_numbers=[]                                                                # Stores the frame numbers
frame_count=0
cap = cv2.VideoCapture(path)                                                    # Opening the video file

if not cap.isOpened():                                                          # Checking if the video is opened successfully
    print("Error: Could not open video.")
    exit()

ret, frame = cap.read()                                                         # Reading the firstframe

if not ret:                                                                     # Checking if the frame was read successfully
    print("Error: Could not read video frame.")
    cap.release()
    exit()

prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                        # Converting the frame to grayscale

while True:
    ret, current_frame = cap.read()                                              # Reading subsequent frames
    if not ret:                                                                  # End of video
        break
    frame_count+=1
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)         # Convertin the current frame to grayscale
    frame_diff = cv2.absdiff(current_frame_gray, prev_frame_gray)                # Taking frame difference for motion detectin
    max_diff = np.max(frame_diff)
    max_frame_diffs.append(max_diff)
    frame_numbers.append(frame_count)
    if max_diff > threshold:                                                     # Condition for motion
        cv2.imshow('Frame', current_frame)                                       # Displaying frames with motion
        if cv2.waitKey(30) & 0xFF == 27:                                         # Press 'Esc' to exit
            break

    prev_frame_gray = current_frame_gray                                         # Setting the current frame as previous frame for the next frame

cap.release()
plt.bar(frame_numbers, max_frame_diffs, color='b')                               # Bar plot of maximum of pixel differences Vs frame number
plt.xlabel('Frame Number')
plt.ylabel('Maximum of Pixel Differences')
plt.title('Maximum of Pixel Differences Between Frames')
plt.show()
#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def detect_motion(path: str, threshold: int) -> tuple[list[int], list[int]]:
    max_frame_diffs = []                                                      # Stores maximum of pixel differences between consecutive frames
    total_frame_diffs = []                                                    # Stores total pixel differences between consecutive frames
    cap = cv2.VideoCapture(path)                                              # Opening the video file

    if not cap.isOpened():                                                    # Checking if the video is opened successfully
        raise FileNotFoundError(f"Could not open {path}.")

    ret, frame = cap.read()                                                   # Reading the first frame

    if not ret:                                                               # Checking if the frame was read successfully
        raise ValueError("Could not read video frame.")

    prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                 # Converting the frame to grayscale

    while True:
        ret, current_frame = cap.read()                                       # Reading subsequent frames
        if not ret:                                                           # End of video
            break
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)  # Converting the current frame to grayscale
        frame_diff = cv2.absdiff(current_frame_gray, prev_frame_gray)         # Taking frame difference for motion detection
        max_diff = np.max(frame_diff)
        max_frame_diffs.append(max_diff)
        total_frame_diffs.append(frame_diff.sum())
        if max_diff > threshold:                                              # Condition for motion
            cv2.imshow('Frame', frame_diff)                                   # Displaying frames with motion
            if cv2.waitKey(30) & 0xFF == 27:                                  # Press 'Esc' to exit
                break

        prev_frame_gray = current_frame_gray                                  # Setting the current frame as previous frame for the next frame

    cap.release()
    cv2.destroyAllWindows()
    return max_frame_diffs, total_frame_diffs

def plot_frame_differences(max_frame_diffs: list[int], total_frame_diffs: list[int]) -> None:
    frame_count = len(max_frame_diffs)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    
    ax1.bar(range(frame_count), total_frame_diffs, color='r')
    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Total Pixel Differences')
    ax1.set_title('Total Pixel Differences Between Frames')
    
    ax2.bar(range(frame_count), max_frame_diffs, color='b')
    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Maximum of Pixel Differences')
    ax2.set_title('Maximum of Pixel Differences Between Frames')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Detect motion in a video file.")
    parser.add_argument("-v", "--video_path", help="Path to the video file", type=str, required=True)
    parser.add_argument("-t", "--threshold", help="Threshold for detecting motion", type=int, default=100)
    parser.add_argument("-p", "--plot", help="Plot the frame differences", action="store_true")
    args = parser.parse_args()
    
    try:
        max_frame_diffs, total_frame_diffs = detect_motion(args.video_path, args.threshold)
        if args.plot:
            plot_frame_differences(max_frame_diffs, total_frame_diffs)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
import cv2
import numpy as np

# Load camera parameters
npz_data = np.load("calibrate.npz")
camera_matrix = npz_data["camera_matrix"]
dist_coeffs = npz_data["dist_coeffs"]

# Open the input video
input_video_path = "output_cam2.mp4"
output_video_path = "output_undistorted.mp4"
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi output

# Create the output video writer
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # Write the frame to the output video
    out.write(undistorted_frame)


# Release everything
cap.release()
out.release()

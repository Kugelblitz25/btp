# Importing the necssary libraries

import cv2
import mediapipe as mp
import numpy as np
from collections import Counter
import webcolors

# Initializing MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_drawing = mp.solutions.drawing_utils

def closest_color(requested_colour):
    '''Function to find the closest color to a requested color from a set of standard colors'''
    min_colours = {}
    for name in webcolors.names("css2"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_color_name(rgb_tuple):
    '''Function to find the color name given an RGB tuple'''
    try:
        # Converting RGB to hex
        hex_value = webcolors.rgb_to_hex(rgb_tuple)
        # Get the color name directly
        return webcolors.hex_to_name(hex_value)
    except ValueError:
        # If exact match not found, finding the closest color
        return closest_color(rgb_tuple)
    
def process_image(frame):
    '''Function to compute key points given the cropped bounding box of the individual 
    and return the face cropped image with the color of apparels worn by the individual ''' 

    ## For testing with an image path
    # frame = cv2.imread(frame)

    # Initializing the mask and gettting frame dimensions
    h, w, _ = frame.shape
    mask = np.zeros((h,w),dtype = np.uint8)

    # Converting frame to RGB for input to MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)

    # Converting back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Processing the results
    if results.pose_landmarks:
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Getting landmarks
        landmarks = results.pose_landmarks.landmark
        right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h))
        left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                        int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h))
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                        int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h))
        right_knee = (int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w),
                int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h))
        left_knee = (int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w),
            int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h))

        # Drawing bounding box for torso
        pts = np.array([right_hip, right_shoulder, left_shoulder, left_hip], dtype=np.int32)  # Ensure the array has int32 type
        pts = pts.reshape((-1, 1, 2))  # Reshaping for cv2.polylines
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)  

        # Creating a mask for the torso region
        cv2.fillPoly(mask, [pts], 255)

        # Extracting the torso region
        torso_region = cv2.bitwise_and(frame, frame, mask)
        torso_pixels = torso_region[np.where(mask==255)]

        # Finding mode color of the torso region
        if len(torso_pixels)>0:
            torso_pixels = torso_pixels.reshape(-1, 3)

            # Finding the mode tuple
            mode_color_torso = Counter(map(tuple, torso_pixels)).most_common(1)[0][0]

            # Converting the mode tuple from BGR to RGB
            torso_mode_color_rgb = (mode_color_torso[2], mode_color_torso[1], mode_color_torso[0])

            # Getting the color name corresponding to the mode tuple
            torso_color_name = get_color_name(torso_mode_color_rgb)
            
            # Defining the lower body region 
            lower_body_pts = np.array([right_hip, left_hip, left_knee, right_knee], dtype=np.int32)
            lower_body_pts = lower_body_pts.reshape((-1, 1, 2))
            cv2.polylines(image, [lower_body_pts], isClosed=True, color=(255, 0, 0), thickness=2)

            # Creating a mask for the lower_body region
            mask.fill(0)  
            cv2.fillPoly(mask, [lower_body_pts], 255)

        # Extracting the lower_body region
        lower_body_region = cv2.bitwise_and(frame, frame, mask=mask)
        lower_body_pixels = lower_body_region[np.where(mask == 255)]

        # Finding mode color of the lower_body region
        if len(lower_body_pixels) > 0:
            lower_body_pixels = lower_body_pixels.reshape(-1, 3)

            # Finding the mode tuple
            lower_body_mode_color = Counter(map(tuple, lower_body_pixels)).most_common(1)[0][0]

            # Converting the mode tuple from BGR to RGB
            lower_body_mode_color_rgb = (lower_body_mode_color[2], lower_body_mode_color[1], lower_body_mode_color[0])

            # Getting the color name corresponding to the mode tuple
            lower_body_color_name = get_color_name(lower_body_mode_color_rgb)

            # Getting the cropped face image
            min_y = min(left_shoulder[1], right_shoulder[1])
            face = frame[0:min_y, :]
            
        return face, torso_color_name, torso_mode_color_rgb, lower_body_color_name, torso_mode_color_rgb, landmarks
        
    else:
        return None
    
## Testing with an image path
# face, torso_color_name, torso_mode_color_rgb, lower_body_color_name, torso_mode_color_rgb, landmarks = process_image("D:/Python_Programs/Miscellaneous/Human-2.png")
# cv2.imshow("Face", face)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()
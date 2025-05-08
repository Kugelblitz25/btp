import numpy as np
import cv2
import os

def calibrate_camera(video_path, checkerboard_size=(9, 6), square_size=1.0, 
                    save_dir='calibration_results', sample_frequency=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size
    
    objpoints = []
    imgpoints = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None, None, None
    
    frame_count = 0
    success_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_frequency == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            
            if ret:
                objpoints.append(objp)
                
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                success_count += 1
                
                vis_frame = frame.copy()
                cv2.drawChessboardCorners(vis_frame, checkerboard_size, corners2, ret)
                cv2.imwrite(os.path.join(save_dir, f'chessboard_detected_{success_count}.jpg'), vis_frame)
        
        frame_count += 1
    
    cap.release()
    
    print(f"Successfully detected checkerboard in {success_count} frames")
    
    if success_count < 10:
        print("Warning: Less than 10 successful detections. Calibration may not be accurate.")
    
    if success_count > 0:
        print("Calibrating camera...")
        
        flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags)
        
        print(f"Calibration complete. RMS reprojection error: {ret}")
        
        # Save all intrinsic and extrinsic parameters to NPZ file
        np.savez(os.path.join(save_dir, 'camera_calibration.npz'), 
                 camera_matrix=mtx, 
                 dist_coeffs=dist, 
                 rvecs=rvecs, 
                 tvecs=tvecs,
                 image_size=gray.shape[::-1],
                 objpoints=objpoints,
                 imgpoints=imgpoints)
        
        with open(os.path.join(save_dir, 'camera_params.txt'), 'w') as f:
            f.write(f"RMS reprojection error: {ret}\n\n")
            f.write("Camera matrix (intrinsics):\n")
            f.write(str(mtx) + "\n\n")
            f.write("Distortion coefficients:\n")
            f.write(str(dist) + "\n")
        
        print(f"Calibration data saved to {save_dir}")
        
        return ret, mtx, dist, rvecs, tvecs, gray.shape[::-1]
    else:
        print("Calibration failed. No checkerboard patterns were detected.")
        return None, None, None, None, None, None

def undistort_video(input_video_path, mtx, dist, output_video_path, image_size):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get optimal new camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, 1, image_size)
    
    # Calculate undistortion maps
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, image_size, cv2.CV_16SC2)
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, image_size)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Undistort the frame
        undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        
        # Crop the undistorted frame
        x, y, w, h = roi
        undistorted = undistorted
        
        # Resize back to original image size if needed
        if (w, h) != image_size:
            undistorted = cv2.resize(undistorted, image_size)
        
        # Write to output video
        out.write(undistorted)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    cap.release()
    out.release()
    print(f"Undistorted video saved to {output_video_path}")

def load_calibration(calibration_file):
    data = np.load(calibration_file)
    return data['camera_matrix'], data['dist_coeffs'], data['image_size']

def main(input_video_path, output_video_path, calibration_file=None, checkerboard_size=(9, 6), square_size=1.0):
    save_dir = 'calibration_results'
    
    if calibration_file and os.path.exists(calibration_file):
        print(f"Loading calibration from {calibration_file}")
        mtx, dist, image_size = load_calibration(calibration_file)
    else:
        print(f"Performing camera calibration using {input_video_path}")
        _, mtx, dist, _, _, image_size = calibrate_camera(
            input_video_path, checkerboard_size, square_size, save_dir)
        
        if mtx is None:
            print("Calibration failed. Cannot undistort video.")
            return
        
        calibration_file = os.path.join(save_dir, 'camera_calibration.npz')
    
    print("Undistorting video...")
    undistort_video(input_video_path, mtx, dist, output_video_path, image_size)
    print("Done!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Camera calibration and video undistortion')
    parser.add_argument('input_video', type=str, help='Input video path')
    parser.add_argument('output_video', type=str, help='Output undistorted video path')
    parser.add_argument('--calibration', type=str, help='Path to existing calibration NPZ file (optional)')
    parser.add_argument('--checkerboard_width', type=int, default=11, help='Number of inner corners along width')
    parser.add_argument('--checkerboard_height', type=int, default=7, help='Number of inner corners along height')
    parser.add_argument('--square_size', type=float, default=5.0, help='Size of checkerboard square in cm')
    
    args = parser.parse_args()
    
    main(
        args.input_video,
        args.output_video,
        args.calibration,
        (args.checkerboard_width, args.checkerboard_height),
        args.square_size
    )
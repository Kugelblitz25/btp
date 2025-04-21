import numpy as np
import cv2
import os

def calibrate_fisheye_camera(video_path, checkerboard_size=(9, 6), square_size=1.0, 
                            save_dir='calibration_results', sample_frequency=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    objp = np.zeros((1, checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size
    
    objpoints = []
    imgpoints = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, None, None, None
    
    frame_count = 0
    success_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_frequency == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, 
                                                    cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                    cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            if ret:
                objpoints.append(objp)
                
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                imgpoints.append(corners2.reshape(1, -1, 2))
                
                success_count += 1
                
                vis_frame = frame.copy()
                cv2.drawChessboardCorners(vis_frame, checkerboard_size, corners2, ret)
                cv2.imwrite(os.path.join(save_dir, f'chessboard_detected_{success_count}.jpg'), vis_frame)
        
        frame_count += 1
    
    cap.release()
    
    print(f"Successfully detected checkerboard in {success_count} frames")
    
    if success_count < 15:
        print("Warning: Less than 15 successful detections. Fisheye calibration may not be accurate.")
    
    if success_count > 0:
        print("Calibrating fisheye camera...")
        
        # Fisheye calibration settings
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW
        
        # Initialize camera matrix
        img_shape = gray.shape[::-1]
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
        
        # Calibrate camera
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            img_shape,
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags
        )
        
        print(f"Fisheye calibration complete. RMS reprojection error: {rms}")
        
        # Save calibration results
        np.savez(os.path.join(save_dir, 'fisheye_calibration_data.npz'), 
                 camera_matrix=K, 
                 dist_coeffs=D, 
                 rvecs=np.array(rvecs), 
                 tvecs=np.array(tvecs))
        
        with open(os.path.join(save_dir, 'fisheye_camera_params.txt'), 'w') as f:
            f.write(f"RMS reprojection error: {rms}\n\n")
            f.write("Camera matrix (K):\n")
            f.write(str(K) + "\n\n")
            f.write("Distortion coefficients (D):\n")
            f.write(str(D) + "\n")
        
        print(f"Fisheye calibration data saved to {save_dir}")
        
        return rms, K, D, np.array(rvecs), np.array(tvecs)
    else:
        print("Calibration failed. No checkerboard patterns were detected.")
        return None, None, None, None

def calibrate_standard_camera(video_path, checkerboard_size=(9, 6), square_size=1.0, 
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
        print("Calibrating camera with standard model...")
        
        # Use 8 distortion coefficients for wide-angle lenses (k1,k2,p1,p2,k3,k4,k5,k6)
        flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags)
        
        print(f"Standard calibration complete. RMS reprojection error: {ret}")
        
        np.savez(os.path.join(save_dir, 'standard_calibration_data.npz'), 
                 camera_matrix=mtx, 
                 dist_coeffs=dist, 
                 rvecs=rvecs, 
                 tvecs=tvecs)
        
        with open(os.path.join(save_dir, 'standard_camera_params.txt'), 'w') as f:
            f.write(f"RMS reprojection error: {ret}\n\n")
            f.write("Camera matrix:\n")
            f.write(str(mtx) + "\n\n")
            f.write("Distortion coefficients (8 parameters for wide-angle):\n")
            f.write(str(dist) + "\n")
        
        print(f"Standard calibration data saved to {save_dir}")
        
        return ret, mtx, dist, rvecs, tvecs
    else:
        print("Calibration failed. No checkerboard patterns were detected.")
        return None, None, None, None, None

def test_undistortion_fisheye(image_path, K, D, save_dir='calibration_results'):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    h, w = img.shape[:2]
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    cv2.imwrite(os.path.join(save_dir, 'fisheye_undistorted.jpg'), undistorted_img)
    cv2.imwrite(os.path.join(save_dir, 'original.jpg'), img)
    
    print(f"Original and fisheye-undistorted images saved to {save_dir}")

def test_undistortion_standard(image_path, mtx, dist, save_dir='calibration_results'):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    h, w = img.shape[:2]
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    cv2.imwrite(os.path.join(save_dir, 'standard_undistorted.jpg'), dst)
    cv2.imwrite(os.path.join(save_dir, 'original.jpg'), img)
    
    print(f"Original and standard-undistorted images saved to {save_dir}")

if __name__ == "__main__":
    video_path = 'output2.mp4'
    checkerboard_size = (11, 7)
    square_size = 2.4
    
    # Try fisheye calibration first
    print("Attempting fisheye calibration (for extreme wide-angle/fisheye lenses)...")
    rms, K, D, _, _ = calibrate_fisheye_camera(
        video_path, checkerboard_size, square_size)
    
    # If fisheye succeeds, test undistortion with fisheye model
    if K is not None:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            test_frame_path = 'calibration_results/test_frame.jpg'
            cv2.imwrite(test_frame_path, frame)
            test_undistortion_fisheye(test_frame_path, K, D)
    
    # Also try standard calibration with extended distortion coefficients
    print("\nAttempting standard calibration with extended coefficients (for wide-angle lenses)...")
    ret, mtx, dist, _, _ = calibrate_standard_camera(
        video_path, checkerboard_size, square_size)
    
    # If standard succeeds, test undistortion with standard model
    if mtx is not None:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            test_frame_path = 'calibration_results/test_frame.jpg'
            cv2.imwrite(test_frame_path, frame)
            test_undistortion_standard(test_frame_path, mtx, dist)
    
    print("\nCalibration complete. Check both fisheye and standard undistortion results to determine which model works better for your camera.")
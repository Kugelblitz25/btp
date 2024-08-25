#!/usr/bin/env python3
from typing import Generator, Optional
import cv2                      # OpenCV library
import numpy as np
import argparse                 # Module for parsing command line arguments
from time import perf_counter   # Function to measure time intervals

algorithms = {                  # Dictionary of different background subtraction methods
        'MOG2': cv2.createBackgroundSubtractorMOG2,
        'KNN': cv2.createBackgroundSubtractorKNN,
        'GMG': cv2.bgsegm.createBackgroundSubtractorGMG,
        'CNT': cv2.bgsegm.createBackgroundSubtractorCNT,
        'GSOC': cv2.bgsegm.createBackgroundSubtractorGSOC,
        'LSBP': cv2.bgsegm.createBackgroundSubtractorLSBP,
        'MOG': cv2.bgsegm.createBackgroundSubtractorMOG,
    }

class Metrics:
    def __init__(self) -> None:
        self.process_time = 0
        self.f1_score = 0
        self.precision = 0
        self.recall = 0
        self.accuracy = 0
        self.count = 0

    def add(self, t: float, f1: float, pre: float, rec: float, acc: float) -> None:
        self.process_time += t
        self.f1_score += f1
        self.precision += pre
        self.recall += rec
        self.accuracy += acc
        self.count += 1

    def avg(self) -> tuple[float, float, float, float, float]:
        if self.count == 0 or self.process_time == 0:
            raise ValueError("Process Time or Number of Frames is zero.")
        
        fps = self.count / self.process_time
        avg_precision = self.precision / self.count
        avg_recall = self.recall / self.count
        avg_f1_score = self.f1_score / self.count
        avg_accuracy = self.accuracy / self.count

        return fps, avg_precision, avg_recall, avg_f1_score, avg_accuracy


class VideoWriter:                              # Class to write frames to an output video file
    def __init__(self, output_path: Optional[str], video_info: tuple[int, int, int]) -> None:                                           # Function to initialize the video writer with the given output path and video information
        self.frame_width, self.frame_height, self.fps = video_info                                                                      # Video information is a tuple of frame width,height and frames per second(fps)
        self.out = None
        self.out = None
        if output_path:
            self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.frame_width*2, self.frame_height)) # Initiaizing the output path
    def write(self, frame: np.ndarray) -> None: # Function to write a single frame to the output path
        if self.out:
            self.out.write(frame)               # Writing the video frame to the output path
    
    def close(self) -> None:                    # Function to finalize the video file
        if self.out:
            self.out.release()                  # Releasing the video writer object, to ensure that the video file is properly saved and closed

class Tester:                                   
    def __init__(self, algorithm: str) -> None: # Function to analyse background subtractor for the given algorithm
        self.algorithm = algorithm              
        try:
            self.bg_subtractor = algorithms[algorithm]()    
        except KeyError:
            raise ValueError(f"Algorithm {algorithm} not available.")
        except AttributeError:
            raise ValueError(f"Algorithm {algorithm} not available in your OpenCV installation.")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 10, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.medianBlur(fg_mask, 3)
        kernel_ex =cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  
        kernel_er =cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) 
        fg_mask = cv2.dilate(fg_mask, kernel_ex, iterations=3)
        fg_mask = cv2.erode(fg_mask, kernel_er, iterations=1)
        return fg_mask
    
    @staticmethod                                                                                                  # Method that belongs to a class rather than any specific instance of that class
    def compare(fg_mask: np.ndarray, ground_truth: Optional[np.ndarray]) -> tuple[np.ndarray, int, int, int, int]: # Function to compare result with ground truth
        if ground_truth is None:                                                                                   # Ground truth is a binary mask of the foreground
            return cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR), 0, 0, 0, 0                                           # Convert the binary mask to colour
        
        fg_mask_color = np.zeros((*fg_mask.shape, 3), dtype=np.uint8)

        true_positives = cv2.bitwise_and(fg_mask, ground_truth)                                                    # Computing true positives
        false_positives = cv2.bitwise_and(fg_mask, cv2.bitwise_not(ground_truth))                                  # Computing false positives
        false_negatives = cv2.bitwise_and(ground_truth, cv2.bitwise_not(fg_mask))                                  # Computing false negatives
        true_negatives = cv2.bitwise_and(cv2.bitwise_not(fg_mask), cv2.bitwise_not(ground_truth))                  # Computing true negatives

        fg_mask_color[true_positives > 0] = [0, 255, 0]                                                            # Colouring the true positives green
        fg_mask_color[false_positives > 0] = [255, 0, 0]                                                           # Colouring the false positives red
        fg_mask_color[false_negatives > 0] = [0, 0, 255]                                                           # Colouring the false negatives blue

        total_tp = true_positives.sum() // 255                                                                     # Computing total true positives
        total_fp = false_positives.sum() // 255                                                                    # Computing total false positives
        total_fn = false_negatives.sum() // 255                                                                    # Computing total false negatives
        total_tn = true_negatives.sum() // 255                                                                     # Computing total true negatives

        if total_fp > 0.9*fg_mask.shape[0]*fg_mask.shape[1]:
            total_tp = total_fp
            total_fp = 0

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0                            # Calculating precision
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0                               # Calculating recall
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0                 # Calculating f1_score
        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)                              # Calculating accuracy

        return fg_mask_color, precision, recall, f1_score, accuracy
    
    @staticmethod
    def load_video(video_path: str, ground_truth_path: Optional[str] = None) -> Generator[np.ndarray, None, None]: # Function to load video frame by frame
        cap = cv2.VideoCapture(video_path)                                                                         # Read frames from the video file
        if not cap.isOpened():
            raise FileNotFoundError(f"Video file {video_path} not found.")
        
        if ground_truth_path:
            gt_cap = cv2.VideoCapture(ground_truth_path)                                                           # Read frames from the ground truth video file
            if not gt_cap.isOpened():
                raise FileNotFoundError(f"Ground truth video file {ground_truth_path} not found.")
        else:
            print("Ground truth video not provided.")
            gt_cap = None
        
        while True:
            ret, frame = cap.read()                                                                               # Reading a single frame from the video
            if not ret:                                                                                           # End of video
                break 

            if gt_cap:
                ret_gt, gt_frame = gt_cap.read()                                                                 # Reading a single frame from the ground truth video
                if not ret_gt:                                                                                   # End of ground truth video
                    break
                ground_truth = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
                _, ground_truth = cv2.threshold(ground_truth, 10, 255, cv2.THRESH_BINARY)
            else:
                ground_truth = None

            yield frame, ground_truth

        cap.release()
        if gt_cap:
            gt_cap.release()

    @staticmethod
    def get_video_info(video_path: str) -> tuple[int, int, int]:                                              # Function to compute video information
        cap = cv2.VideoCapture(video_path)                                                                    # Read frames from the video file
        if not cap.isOpened():
            raise FileNotFoundError(f"Video file {video_path} not found.")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                                                 # Getting frame width
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                                               # Getting frame height
        fps = int(cap.get(cv2.CAP_PROP_FPS))                                                                 # Getting frames per second
        cap.release()

        return frame_width, frame_height, fps

    def test(self, 
             video_path: str, 
             ground_truth_path: Optional[str] = None, 
             output_path: Optional[str] = None,
             print_results: Optional[bool] = False,
             show_fg: Optional[bool] = False) -> tuple[float, float, float, float]:
        video_info = self.get_video_info(video_path)                                                        # Get video information
        writer = VideoWriter(output_path, video_info)                                                       # Write video to the output path
        metrics = Metrics()
        try:
            for frame, ground_truth in self.load_video(video_path, ground_truth_path):
                t1 = perf_counter()                                                                         # Start time
                fg_mask = self.apply(frame)                                                                 # Compute and apply foreground mask
                t2 = perf_counter()                                                                         # End time
                fg_mask_color, precision, recall, f1, accuracy = self.compare(fg_mask, ground_truth)        # Comparing result with ground truth
                
                metrics.add(t2-t1, f1, precision, recall, accuracy)

                if show_fg:
                    fg_mask_color = cv2.bitwise_and(frame, frame, mask = fg_mask)

                output_frame = np.hstack([frame, fg_mask_color])
                writer.write(output_frame)
                cv2.imshow('Frame', output_frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"An error occurred during testing: {str(e)}")
        finally:
            writer.close()
            cv2.destroyAllWindows()
        
        fps, avg_precision, avg_recall, avg_f1, avg_accuracy =  metrics.avg()

        if print_results:                         # Printing results
            print()
            print("-" * 118)
            print(f"|{'Video':^25}|{'Ground Truth':^25}|{'Precision':^15}|{'Recall':^15}|{'F1 Score':^10}|{'Accuracy':^10}|{'FPS':^10}|")
            print("-" * 118)
            print(f"|{video_path:^25}|{ground_truth_path or '':^25}|{avg_precision:^15.3f}|{avg_recall:^15.3f}|{avg_f1:^10.3f}|{avg_accuracy:^10.3f}|{fps:^10.3f}|")
            print("-" * 118)
            print()

        return fps, avg_precision, avg_recall, avg_f1, avg_accuracy

    def __str__(self) -> str:
        return f"Tester object for {self.algorithm} algorithm."
    

def main() -> None:
    parser = argparse.ArgumentParser(description='Apply background subtraction algorithms to a video.')
    parser.add_argument('-v', '--video_path', type=str, help='Path to input video file.', required=True)
    parser.add_argument('-a', '--algorithm', type=str, help='Background subtraction algorithm to use.', default='MOG2')
    parser.add_argument('-gt', '--ground_truth', type=str, help='Path to ground truth video file.', default=None)
    parser.add_argument('-o', '--output', type=str, help='Path to output video file.', default=None)
    parser.add_argument('-fg', '--foreground', type=bool, help='Path to output video file.', default=False)

    args = parser.parse_args()

    tester = Tester(args.algorithm.upper())
    tester.test(args.video_path, args.ground_truth, args.output, True, args.foreground)

if __name__ == '__main__':
    main()
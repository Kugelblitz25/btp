#!/usr/bin/env python3
from typing import Generator, Optional
import cv2
import numpy as np
import argparse
from time import perf_counter

algorithms = {
        'MOG2': cv2.createBackgroundSubtractorMOG2,
        'KNN': cv2.createBackgroundSubtractorKNN,
        'GMG': cv2.bgsegm.createBackgroundSubtractorGMG,
        'CNT': cv2.bgsegm.createBackgroundSubtractorCNT,
        'GSOC': cv2.bgsegm.createBackgroundSubtractorGSOC,
        'LSBP': cv2.bgsegm.createBackgroundSubtractorLSBP,
        'MOG': cv2.bgsegm.createBackgroundSubtractorMOG,
    }

class VideoWriter:
    def __init__(self, output_path: Optional[str], video_info: tuple[int, int, int]) -> None:
        self.frame_width, self.frame_height, self.fps = video_info
        self.out = None
        if output_path:
            self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.frame_width, self.frame_height))
    
    def write(self, frame: np.ndarray) -> None:
        if self.out:
            self.out.write(frame)
    
    def close(self) -> None:
        if self.out:
            self.out.release()

class Tester:
    def __init__(self, algorithm: str) -> None:
        self.algorithm = algorithm
        try:
            self.bg_subtractor = algorithms[algorithm]()
        except KeyError:
            raise ValueError(f"Algorithm {algorithm} not available.")
        except AttributeError:
            raise ValueError(f"Algorithm {algorithm} not available in your OpenCV installation.")

    def apply(self, frame: np.ndarray) -> np.ndarray:
        fg_mask = self.bg_subtractor.apply(frame)
        return fg_mask
    
    @staticmethod
    def compare(fg_mask: np.ndarray, ground_truth: Optional[np.ndarray]) -> tuple[np.ndarray, int, int, int, int]:
        if ground_truth is None:
            return cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR), 0, 0, 0, 0
        
        fg_mask_color = np.zeros((*fg_mask.shape[:2], 3), dtype=np.uint8)

        true_positives = cv2.bitwise_and(fg_mask, ground_truth)
        false_positives = cv2.bitwise_and(fg_mask, cv2.bitwise_not(ground_truth))
        false_negatives = cv2.bitwise_and(ground_truth, cv2.bitwise_not(fg_mask))
        true_negatives = cv2.bitwise_and(cv2.bitwise_not(fg_mask), cv2.bitwise_not(ground_truth))

        fg_mask_color[true_positives > 0] = [0, 255, 0]
        fg_mask_color[false_positives > 0] = [255, 0, 0]
        fg_mask_color[false_negatives > 0] = [0, 0, 255]

        total_tp = true_positives.sum() // 255
        total_fp = false_positives.sum() // 255
        total_fn = false_negatives.sum() // 255
        total_tn = true_negatives.sum() // 255

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) 

        return fg_mask_color, precision, recall, f1_score, accuracy
    
    @staticmethod
    def load_video(video_path: str, ground_truth_path: Optional[str] = None) -> Generator[np.ndarray, None, None]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video file {video_path} not found.")
        
        if ground_truth_path:
            gt_cap = cv2.VideoCapture(ground_truth_path)
            if not gt_cap.isOpened():
                raise FileNotFoundError(f"Ground truth video file {ground_truth_path} not found.")
        else:
            print("Ground truth video not provided.")
            gt_cap = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if gt_cap:
                ret_gt, gt_frame = gt_cap.read()
                if not ret_gt:
                    break
                ground_truth = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
            else:
                ground_truth = None

            yield frame, ground_truth

        cap.release()
        if gt_cap:
            gt_cap.release()

    @staticmethod
    def get_video_info(video_path: str) -> tuple[int, int, int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video file {video_path} not found.")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        return frame_width, frame_height, fps

    def test(self, 
             video_path: str, 
             ground_truth_path: Optional[str] = None, 
             output_path: Optional[str] = None,
             print_results: Optional[bool] = False) -> tuple[float, float, float, float]:
        video_info = self.get_video_info(video_path)
        writer = VideoWriter(output_path, video_info)
        precision_sum, recall_sum, f1_sum, accuracy_sum, num_frames = 0, 0, 0, 0, 0
        tot_time = 0
        try:
            for frame, ground_truth in self.load_video(video_path, ground_truth_path):
                t1 = perf_counter()
                fg_mask = self.apply(frame)
                t2 = perf_counter()
                tot_time += t2 - t1
                fg_mask_color, precision, recall, f1, accuracy = self.compare(fg_mask, ground_truth)
                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
                accuracy_sum += accuracy
                num_frames += 1
                writer.write(fg_mask_color)
                cv2.imshow('Frame', fg_mask_color)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"An error occurred during testing: {str(e)}")
        finally:
            writer.close()
            cv2.destroyAllWindows()

        avg_precision = precision_sum / num_frames 
        avg_recall = recall_sum / num_frames 
        avg_f1 = f1_sum / num_frames
        avg_accuracy = accuracy_sum / num_frames
        avg_time = tot_time / num_frames
        avg_fps = 1 / avg_time

        if print_results:
            print()
            print("-" * 118)
            print(f"|{'Video':^25}|{'Ground Truth':^25}|{'Precision':^15}|{'Recall':^15}|{'F1 Score':^10}|{'Accuracy':^10}|{'FPS':^10}|")
            print("-" * 118)
            print(f"|{video_path:^25}|{ground_truth_path:^25}|{avg_precision:^15.3f}|{avg_recall:^15.3f}|{avg_f1:^10.3f}|{avg_accuracy:^10.3f}|{avg_fps:^10.3f}|")
            print("-" * 118)
            print()

        return avg_precision, avg_recall, avg_f1, avg_accuracy

    def __str__(self) -> str:
        return f"Tester object for {self.algorithm} algorithm."
    

def main() -> None:
    parser = argparse.ArgumentParser(description='Apply background subtraction algorithms to a video.')
    parser.add_argument('-v', '--video_path', type=str, help='Path to input video file.', required=True)
    parser.add_argument('-a', '--algorithm', type=str, help='Background subtraction algorithm to use.', default='MOG2')
    parser.add_argument('-gt', '--ground_truth', type=str, help='Path to ground truth video file.', default=None)
    parser.add_argument('-o', '--output', type=str, help='Path to output video file.', default=None)
    args = parser.parse_args()

    tester = Tester(args.algorithm.upper())
    tester.test(args.video_path, args.ground_truth, args.output, True)

if __name__ == '__main__':
    main()
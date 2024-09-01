#!/usr/bin/env python3
from typing import Generator, Optional
import cv2                                  # OpenCV library
import numpy as np
import argparse                             # Module for parsing command line arguments
from time import perf_counter               # Function to measure time intervals

algorithms: dict[str, callable] = {         # Dictionary of different background subtraction methods
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
        self.process_time = 0.0
        self.f1_score = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.accuracy = 0.0
        self.count = 0

    def score(self, t: float, fg_mask: np.ndarray, ground_truth: Optional[np.ndarray]) -> None:
        if ground_truth is None:
            return  # No ground truth provided
        fg_mask = fg_mask // 255
        ground_truth = ground_truth // 255

        TP = np.bitwise_and(fg_mask, ground_truth).sum()
        FP = np.bitwise_and(fg_mask, np.bitwise_not(ground_truth)).sum()
        FN = np.bitwise_and(np.bitwise_not(fg_mask), ground_truth).sum()
        TN = np.bitwise_and(np.bitwise_not(fg_mask), np.bitwise_not(ground_truth)).sum()

        if FP > 0.9 * fg_mask.size:
            return

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0

        self.process_time += t
        self.f1_score += f1_score
        self.precision += precision
        self.recall += recall
        self.accuracy += accuracy
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


class VideoWriter:
    def __init__(self, output_path: Optional[str], video_info: tuple[int, int, int]) -> None:
        self.frame_width, self.frame_height, self.fps = video_info
        self.out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.frame_width * 2, self.frame_height))

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
        _, fg_mask = cv2.threshold(fg_mask, 10, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.medianBlur(fg_mask, 3)
        kernel_ex = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.dilate(fg_mask, kernel_ex, iterations=3)
        fg_mask = cv2.erode(fg_mask, kernel_er, iterations=1)
        return fg_mask

    @staticmethod
    def colorize(fg_mask: np.ndarray, ground_truth: Optional[np.ndarray]) -> np.ndarray:
        if ground_truth is None:
            return cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        fg_mask_color = np.zeros((*fg_mask.shape, 3), dtype=np.uint8)

        both_positives = cv2.bitwise_and(fg_mask, ground_truth)
        false_positives = cv2.bitwise_and(fg_mask, cv2.bitwise_not(ground_truth))
        false_negatives = cv2.bitwise_and(ground_truth, cv2.bitwise_not(fg_mask))

        fg_mask_color[both_positives > 0] = [0, 255, 0]     # Green for true positives
        fg_mask_color[false_positives > 0] = [255, 0, 0]    # Red for false positives
        fg_mask_color[false_negatives > 0] = [0, 0, 255]    # Blue for false negatives

        return fg_mask_color

    @staticmethod
    def load_video(video_path: str, ground_truth_path: Optional[str] = None) -> Generator[tuple[np.ndarray, Optional[np.ndarray]], None, None]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video file {video_path} not found.")

        gt_cap = None
        if ground_truth_path:
            gt_cap = cv2.VideoCapture(ground_truth_path)
            if not gt_cap.isOpened():
                raise FileNotFoundError(f"Ground truth video file {ground_truth_path} not found.")
        else:
            print("Ground truth video not provided.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ground_truth = None
                if gt_cap:
                    ret_gt, gt_frame = gt_cap.read()
                    if not ret_gt:
                        break
                    ground_truth = cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY)
                    _, ground_truth = cv2.threshold(ground_truth, 10, 255, cv2.THRESH_BINARY)

                yield frame, ground_truth
        finally:
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
             print_results: bool = False,
             show_fg: bool = False) -> tuple[float, float, float, float, float]:
        video_info = self.get_video_info(video_path)
        writer = VideoWriter(output_path, video_info)
        metrics = Metrics()

        try:
            for frame, ground_truth in self.load_video(video_path, ground_truth_path):
                t1 = perf_counter()
                fg_mask = self.apply(frame)
                t2 = perf_counter()
                fg_mask_color = self.colorize(fg_mask, ground_truth)

                metrics.score(t2 - t1, fg_mask, ground_truth)

                if show_fg:
                    fg_mask_color = cv2.bitwise_and(frame, frame, mask=fg_mask)

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

        fps, avg_precision, avg_recall, avg_f1, avg_accuracy = metrics.avg()

        if print_results:
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
    parser.add_argument('-fg', '--foreground', action='store_true', help='Show foreground mask instead of colorized output.')

    args = parser.parse_args()

    tester = Tester(args.algorithm.upper())
    tester.test(args.video_path, args.ground_truth, args.output, True, args.foreground)


if __name__ == '__main__':
    main()
# Background Subtraction Video Processing

This script applies various background subtraction algorithms to a video and optionally compares the results with a ground truth video.

## Installation

1. Ensure you have Python 3 installed.
2. Create a virtual environment.

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install the required dependencies using pip:

    ```bash
    pip install opencv-contrib-python
    ```

## Usage

To run the script, use the following command:

```bash
python3 bgsep/algoTester.py -v <video_path> -a <algorithm> [-gt <ground_truth_path>] [-o <output_path>]
```

### Arguments

- -v, --video_path: Path to the input video file. (Required)
- -a, --algorithm: Background subtraction algorithm to use. Options are MOG2, KNN, GMG, CNT, GSOC, LSBP, MOG. Default is MOG2.
- -gt, --ground_truth: Path to the ground truth video file. (Optional)
- -o, --output: Path to the output video file. (Optional)

## Classes and Methods

`VideoWriter`

- `__init__(self, output_path: Optional[str], video_info: tuple[int, int, int])`: Initializes the video writer.
- `write(self, frame: np.ndarray) -> None`: Writes a frame to the output video.
- `close(self) -> None`: Releases the video writer.

`Tester`

- `__init__(self, algorithm: str) -> None`: Initializes the tester with the specified algorithm.
- `apply(self, frame: np.ndarray) -> np.ndarray`: Applies the background subtraction algorithm to a frame.
- `compare(fg_mask: np.ndarray, ground_truth: Optional[np.ndarray]) -> tuple[np.ndarray, int, int, int, int]`: Compares the foreground mask with the ground truth.
- `load_video(video_path: str, ground_truth_path: Optional[str] = None) -> Generator[np.ndarray, None, None]`: Loads the video and optionally the ground truth video.
- `get_video_info(video_path: str) -> tuple[int, int, int]`: Retrieves the video information (width, height, fps).
- `test(self, video_path: str, ground_truth_path: Optional[str] = None, output_path: Optional[str] = None, print_results: Optional[bool] = False) -> tuple[float, float, float, float]`: Tests the algorithm on the video and optionally compares with the ground truth.

## Example

To test the script with the MOG2 algorithm and save the output:

```bash
python3 bgsep/algoTester.py -v input.mp4 -a MOG2 -o output.mp4
```

To test the script with the KNN algorithm and compare with a ground truth video:

```bash
python3 bgsep/algoTester.py -v input.mp4 -a KNN -o output.mp4
```

## Output

```bash
python3 bgsep/algoTester.py -v videos/vid1.mp4 -gt videos/vid1-true.mp4 -a knn

----------------------------------------------------------------------------------------------------------------------
|            Video             |         Ground Truth         | Avg. FP% | Avg. FN% | F1 Score | Accuracy |   FPS    |
----------------------------------------------------------------------------------------------------------------------
|       videos/vid1.mp4        |     videos/vid1-true.mp4     |  0.396   |  0.146   |  0.613   |  0.995   | 204.889  |
----------------------------------------------------------------------------------------------------------------------

```

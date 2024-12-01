# To utilize GPU for MediaPipe 
  ```shell
 set MEDIAPIPE_DISABLE_GPU=0
  ```
# Model weights for action classification

You can download the Modified ResNet50 weights here  : [Google Drive](https://drive.google.com/file/d/17cIIypzvXRlMo1fVcDBh2rGQVEsU5txx/view?usp=sharing)

You can download the Modified ViT weights here  : [Google Drive](https://drive.google.com/file/d/10vCDEKm_epNsTqfxxRsrliW22V0lfKTz/view?usp=sharing)

# FairMOT
Multi-object tracking 

Steps to run the model for an input video:

* Clone the repository
* Move to the FairMOT directory :
  ```shell
  cd FairMOT
  ```
* Install the requirements :
  ```shell
  pip install -r "requirements.txt"
  ```
* Move to the DCNv2 directory in FairMOT : 
    ```shell
   cd FairMOT
  ```
* Run setup.py :
  ```shell
  python setup.py build develop
  ``` 
* Move to the FairMOT directory :
  ``` shell
  cd ..
  ```
* Run the model for you input video by providing your video path and the desirable parameters :
  ```shell
   python src/demo.py mot --load_model models/fairmot_dla34.pth --conf_thres 0.4 --input-video 'your_video_path.mp4'
  ```
* The output video will be written to a demos folder

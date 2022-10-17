# DMS_v2
## Introdution
We add the **registor** and **face recognition** function into [DMS_v1](https://github.com/qwe12345113/DMS_v1).

In **registor** function, we need to collect 15 photos to build the user database. In **face recognition** function, take a photo then function will recognize the user.

We use the face landmarks to detect the 4 of driving behaviors, including **yawn**, **distraction**, **lower head**, and **closing eyes**.

## Requirements
### Dependencies
- gcc >= 4.7
- cmake >= 3.1
- opencv >= 3.4.14
- dlib == 19.24

### It was tested and runs under the following OSs:
- ubuntu 20.04

Might work under others, but didn't get to test any other OSs just yet.

## Preparing
1. Download [Dlib](http://dlib.net/) and the following model：
    - [face_recognition_resnet_model](https://github.com/davisking/dlib-models/blob/master/dlib_face_recognition_resnet_model_v1.dat.bz2)
    - [68-Dlib's point model](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)
    - [5-Dlib's point model](https://github.com/davisking/dlib-models/blob/master/shape_predictor_5_face_landmarks.dat.bz2)
    

2. Extract the dlib and model, place them into DMS_v2. 
3. Structure of this project should be：
```
DMS_v2
  ├─ src
  ├─ dlib-19.24
  └─ Model
      ├─ shape_predictor_68_face_landmarks.dat
      ├─ shape_predictor_5_face_landmarks.dat
      └─ dlib_face_recognition_resnet_model_v1.dat
```

## Compile
    $ mkdir build && cd build && cmake .. && make -j4

## Getting Started:
### Usage
* detect from image.
```bash
$ ./dms pic /path/to/image
```
* detect from video.
```bash
$ ./dms video /path/to/video
```
* detect from webcam.
```bash
$ ./dms
```
* registor.(press Enter to take photos)
```bash
$ ./dms user_name
```
* face recognition.(press Enter to take a photo)
```bash
$ ./dms rec
```
## Demo


https://user-images.githubusercontent.com/11375811/195013024-d6e1a8ff-211d-40e0-93b6-3ed299b036d2.mp4



# Reference
[1] Tutorial 1：<https://www.learnopencv.com/facemark-facial-landmark-detection-using-opencv/>

[2] Tutorial 2：<https://github.com/spmallick/learnopencv/tree/master/FacialLandmarkDetection>

[3] Tutorial 3：<https://blog.csdn.net/u013841196/article/details/85041007>

[4] Tutorial 4：<https://github.com/e-candeloro/Driver-State-Detection>

[5] Yawn Detection：<https://github.com/deveshdatwani/facial-expression-recognition>

[6] Drowsiness detection：<https://pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/>

[7] Real-Time Eye Blink Detection using Facial Landmarks：<http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf>

[8] Head Pose Estimattion：<https://blog.csdn.net/cdknight_happy/article/details/79975060>

[9] OpenCV Head Pose computation overview：<https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html>

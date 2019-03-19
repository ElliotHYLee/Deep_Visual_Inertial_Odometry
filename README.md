# VisualOdometry3D

Using convolutional neural network, the velocity of the camera is estimated. 
After predicting the velocity, 3D transformation matrices are concanated to estimate the position.

[Click for Youtube video:
<img src="https://github.com/ElliotHYLee/OpticalFlow2VelocityEstimation/blob/master/Images/Capture.JPG">](https://youtu.be/-t8VCICzGD0)

Citation: <br>Hongyun Lee, Matthew McCrink, and James W. Gregory. "Visual-Inertial Odometry for Unmanned Aerial Vehicle using Deep Leering", 2019 Intelligent/Autonomous Guidance and Navigation, AIAA SciTech Conference, accepted for publication


## References(current & future)
- DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks(https://senwang.gitlab.io/DeepVO/files/wang2017DeepVO.pdf)


## ToDo
- LSTM integration
- upload weight.pt 
- explain data set & location

## Prereq.s
<pre>
pip install numpy
pip install scipy
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install pathlib
pip install pypng
pip install pillow
pip install django
pip install image
pip install opencv-python opencv-contrib-python
</pre>
detail: https://github.com/ElliotHYLee/SystemReady


## Tested System
- Hardware
<pre>
CPU: i9-7940x
RAM: 128G, 3200Hz
GPU: two Gefore 1080 ti
MB: ROG STRIX x299-E Gaming
</pre>

- Software
<pre>
Windows10
Python3 with native pip
PyTorch: v1
CUDA: v9.0
Cudnn: v7.1
</pre>
## Run

```
python main_cnn.py
```



## Quick Overview of The Results

<ul>
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr5.png" width="400">
</ul>

## Traing Results
description

<ul>
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq0_pos.png" width="400">
</ul>

## Test Results
<ul>
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq1_pos.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq3_pos.png" width="400">
</ul>

## Correction Result

<ul>
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr10.png" width="400">
</ul>

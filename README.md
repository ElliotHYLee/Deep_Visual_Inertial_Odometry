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
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr8.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr9.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr10.png" width="400">
</ul>


## Traing Results

description

<ul>
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq0_pos.png" width="400">
</ul>

## Test Results

The first three predictions' mse = [0.611, 0.726, 1.162] meters per frame. One cause can be the texture-lacking environments of the sequence 1,3, and 4. Anouther is that these sequences have only a few hundreds data points. Even though these are trained, the generalization is limited. <br>

Yet, if the model is fed with "similar" environement, mse is not too irrational. For the rest of the sets, mse = [0.381 0.419 0.461 0.632]

<ul>
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq1_pos.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq3_pos.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq5_pos.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq7_pos.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq8_pos.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq9_pos.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq10_pos.png" width="400">
</ul>

## Covariance Estimation Without Correction
First a few covariance ellipses are plotted. As the time goes, the covariance eventually blows up.
<ul>
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq0_cov.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq1_cov.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq2_cov.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq3_cov.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq4_cov.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq5_cov.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq6_cov.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq7_cov.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq8_cov.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq9_cov.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq10_cov.png" width="400">
</ul>


## Correction Result

<ul>
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr0.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr1.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr2.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr3.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr4.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr5.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr6.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr7.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr8.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr9.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr10.png" width="400">
</ul>

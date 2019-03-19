# VisualOdometry3D
# VisualOdometry

Using convolutional neural network, the velocity of the camera is estimated. After predicting the velocity, it is numerically integrated to calculate the positions of the camera. <br>


[Click for Youtube video:
<img src="https://github.com/ElliotHYLee/OpticalFlow2VelocityEstimation/blob/master/Images/Capture.JPG">](https://youtu.be/-t8VCICzGD0)

Citation: <br>Hongyun Lee, Matthew McCrink, and James W. Gregory. "Visual-Inertial Odometry for Unmanned Aerial Vehicle using Deep Leering", 2019 Intelligent/Autonomous Guidance and Navigation, AIAA SciTech Conference, accepted for publication


## ToDo
1. Mapping

## Prereq.s

Tensorflow version: 1.3.1 (Google Tensorflow .whl for 1.3.1)

OpenCV 3.2 for data generation.

*** Only tested with TF 1.3.1 GPU and OpenCV GPU builds. GTX1080ti, ubuntu16.

python libraries:

```
chmod +x pythonReady.sh
yes "yes" | sudo sh pythonReady.sh
```
## Run

```
python main.py 0
```

## Quick Overview of The Results

<ul>
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr5.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr8.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr9.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr10.png" width="400">
</ul>

## References(current & future)

- FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks(https://arxiv.org/abs/1612.01925)
- DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks(https://senwang.gitlab.io/DeepVO/files/wang2017DeepVO.pdf)

## Traing Results

description

<ul>
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq0_pos.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq2_pos.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq4_pos.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq6_pos.png" width="400">
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

# Deep Visual Inertial Odometry

Deep learning based visual-inertial odometry project.
<br>
pros:
- Lighter CNN structure. No RNNs -> much lighter.
- Training images together with inertial data using exponential mapping.
- No RNN but Kalman filter: robust rotational prediction.
- Accleration and image fusion for frame-to-frame displacement.

cons:
- no position correction: drift in position: But SLAM can correct the position drfit.


## Please Cite:
Hongyun Lee, Matthew McCrink, and James W. Gregory. "Visual-Inertial Odometry for Unmanned Aerial Vehicle using Deep Leering", 2019 Intelligent/Autonomous Guidance and Navigation, AIAA SciTech Conference, https://doi.org/10.2514/6.2019-1410


## References(current & future)
Please see paper.

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
Python3
PyTorch: v1
CUDA: v9.0
Cudnn: v7.1
</pre>
## Run

```
python main_cnn.py
```

## Traing Results
description

<ul>
<img src="https://github.com/ElliotHYLee/VisualOdometry3D/blob/master/Results/Figures/master_airsim_mr0_results.png" width="400">
</ul>

## Test Results
<ul>
<img src="https://github.com/ElliotHYLee/VisualOdometry3D/blob/master/Results/Figures/master_airsim_mr1_results.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry3D/blob/master/Results/Figures/master_airsim_mr2_results.png" width="400">
</ul>

## Correction Result

<ul>
<img src="https://github.com/ElliotHYLee/VisualOdometry/blob/master/Results/Images/seq_corr10.png" width="400">
</ul>

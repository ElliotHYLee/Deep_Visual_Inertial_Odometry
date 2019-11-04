# Deep Visual Inertial Odometry

Deep learning based visual-inertial odometry project.
<br>
pros:
- Lighter CNN structure. No RNNs -> much lighter.
- Training images together with inertial data using exponential mapping.
- Rotation is coming from external attitude estimation.
- No RNN but Kalman filter: Accleration and image fusion for frame-to-frame displacement.

cons:
- no position correction: drift in position: But SLAM can correct the position drfit.


## Please Cite:
Hongyun Lee, Matthew McCrink, and James W. Gregory. "Deep Learning for Visual-Inertial Odometry: Estimation of Monocular Camera Ego-Motion and its Uncertainty" The Ohio State University, Master Thesis, https://etd.ohiolink.edu/pg_10?0::NO:10:P10_ACCESSION_NUM:osu156331321922759


## References(current & future)
Please see paper.

## Usage:
0. git clone -- recursive https://github.com/ElliotHYLee/Deep_Visual_Inertial_Odometry 
1. Put the .m (Matlab) files under KITTI/odom/dataset/. The files are at DataGenerator folder.
2. run make_trainable_data.m
3. In src/Parampy, change the path for KITTI.
4. At Deep_Visual_Inertial_Odometry, "python main.py"


## ToDo
- upload weight.pt
- change Matlab data get to python 


## Prereq.s
1. Matlab
2. Python 3.5
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
<img src="https://github.com/ElliotHYLee/Deep_Visual_Inertial_Odometry/blob/master/Results/Figures/master_kitti_none0_results.png" width="400">
</ul>

## Test Results
<ul>
<img src="https://github.com/ElliotHYLee/Deep_Visual_Inertial_Odometry/blob/master/Results/Figures/master_kitti_none5_results.png" width="400">
<img src="https://github.com/ElliotHYLee/VisualOdometry3D/blob/master/Results/Figures/master_airsim_mr2_results.png" width="400">
</ul>

## Correction Result

<ul>
<img src="" width="400">
</ul>

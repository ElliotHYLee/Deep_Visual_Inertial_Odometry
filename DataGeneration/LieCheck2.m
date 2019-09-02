clc, clear, close all

euler1 = [0 0 0]*pi/180
euler2 = [0 0 0]*pi/180

R1 = eul2rotm(euler1, 'zyx')
R2 = eul2rotm(euler2, 'zyx')

dR = R1'*R2
th = acos((trace(dR)-1)/2)
skew = th/(2*sin(th))*(dR-dR')
w = [-skew(2,3), skew(1,3), -skew(1,2)]

th = sqrt(w*w')
dR = eye(3) + sin(th)/th*skew + (1-cos(th))/th^2*skew*skew




















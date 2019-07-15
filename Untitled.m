clc, clear, close all

A = [-1 1 0;
     1 1 0;
     0 0 0]
 
[vec, val] = eig(A)

R = eul2rotm([0 0 45*pi/180])





























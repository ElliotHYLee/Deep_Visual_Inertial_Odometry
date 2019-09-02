clc, clear, close all

lie = Lie();
se3 = LieSE3();
so3 = LieSO3();

T00 = [eye(3),[0 0 0]';zeros(1,3),1]
R1 = eul2rotm([45, 0, 0]*pi/180, 'zyx');
T01 = [R1,[1 1 0]';zeros(1,3),1]
R2 = eul2rotm([90, 0, 0]*pi/180, 'zyx');
T02 = [R2,[2 2 0]';zeros(1,3),1]

T12 = T01^-1*T02
[w,u] = se3.getLog(T12);
dT = se3.getExp(w,u);




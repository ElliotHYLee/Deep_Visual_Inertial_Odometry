clc, clear, close all
dsName = 'kitti';
subType = 'none';
seq = 0;

loadData


for i =1:1:N
   rotm = reshape(linR(i,:), 3,3)';
   dtr_Q_gnd{i} = rotm*dtr_Q{i}*rotm';
   cov3(i,:) = diag(dtr_Q_gnd{i});
end
dtr_gnd_std3 = sqrt(cov3);

%%
pos = [0 0 0];
A = eye(3);
H = eye(3);
P{1} = eye(3)*10^-5;
for i=1:1:N
    pos(i+1,:) = pos(i,:) + pr_dtr_gnd(i,:);
    P{i+1} = A*P{i}*A' + dtr_Q_gnd{i};%dtr_Q_gnd{i};
    xx = dtr_Q_gnd{i};
    [vec, val] = eig(xx);
end

plotELPS(dsName, gt_pos, pr_pos, P)
















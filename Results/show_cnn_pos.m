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
R = eye(3)*10
for i=1:1:N
    pos(i+1,:) = pos(i,:) + pr_dtr_gnd(i,:);
    P{i+1} = A*P{i}*A' + dtr_Q_gnd{i};
end

pos_int = cumtrapz(pr_dtr_gnd);


plotELPS(dsName, gt_pos, pos_int, P)


for i=1:1:N
    posCov3(i,:) = diag(P{i});
end

dPos_std = sqrt(posCov3);

% figure
% hold on
% plot(dPos_std(:,1), 'r.')
% plot(dPos_std(:,2), 'g.')
% plot(dPos_std(:,3), 'b.')
% % ylim([0 1])
% 
% figure
% hold on
% plot(dtr_gnd_std3(:,1), 'r.')
% plot(dtr_gnd_std3(:,2), 'g.')
% plot(dtr_gnd_std3(:,3), 'b.')
% % ylim([0 1])

















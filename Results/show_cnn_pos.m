clc, clear, close all
dsName = 'airsim';
subType = 'mrseg';
seq = 2;
loadData


for i =1:1:N
   rotm = reshape(linR(i,:), 3,3)';
%    angle = rotm2eul(rotm);
%    eul(i,:) = [angle(3), angle(2), angle(1)];
%    rotm = eul2rotm(eul(i,:), 'xyz');   
   dtr_Q_gnd{i} = rotm*dtr_Q{i}*rotm';
   cov3(i,:) = diag(dtr_Q_gnd{i});
end
dtr_gnd_std3 = sqrt(cov3);

% figure
% subplot(3,1,1)
% plot(eul(:,1))
% subplot(3,1,2)
% plot(eul(:,2))
% subplot(3,1,3)
% plot(eul(:,3))

figure
subplot(3,1,1)
plot(gt_dw(:,1))
subplot(3,1,2)
plot(gt_dw(:,2))
subplot(3,1,3)
plot(gt_dw(:,3))


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
    
    ang = rotm2eul(vec)*180/pi;
    
%     
%     if i==10
%         asd
%     end
end

plotELPS(dsName, gt_pos, pr_pos, P)


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

figure
subplot(3,1,1)
plot(dtr_gnd_std3(:,1))
subplot(3,1,2)
plot(dtr_gnd_std3(:,2))
subplot(3,1,3)
plot(dtr_gnd_std3(:,3))

% ylim([0 1])

















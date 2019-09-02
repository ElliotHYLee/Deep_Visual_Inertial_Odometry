clc, clear, close all

dsName = 'kitti';
subType = 'none';
seq = 5;
noises=[0 2 16];

posKF = {};
for ii =1:1:3
    noise = noises(ii)
    loadData;
    posGT = gt_pos;
    posKF{ii,1} = kf_pos;
    
end

figure
hold on
plot(posGT(:,1), posGT(:,3), 'r')
plot(posKF{1}(:,1), posKF{1}(:,3), 'g')
plot(posKF{2}(:,1), posKF{2}(:,3), 'b')
plot(posKF{3}(:,1), posKF{3}(:,3), 'k.')
xlim([-300 400])
ylim([-100 500])
lgd = legend('GT', '0 %', '2 %', '16 %')
lgd.FontSize = 20;
xlabel(['\fontsize{20} Position_x, m'])
ylabel(['\fontsize{20} Position_z, m'])






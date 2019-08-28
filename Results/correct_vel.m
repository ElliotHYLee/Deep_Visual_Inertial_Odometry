clc, clear, close all
dsName = 'kitti';
subType = 'none';
seq = 5;
noise=0;
loadData

time = cumtrapz(dt);
acc_gnd = dt.*acc_gnd;
vel_imu = cumtrapz(time, acc_gnd);

% KF5
velKF = [0 0 0];
A = eye(3);
H = eye(3);
P{1} = eye(3)*10^-10;
R = [1 0 0; 0 1 0; 0 0 1]*10^-5

%airsim mr
% R = [4.4016473e-04 -1.0391317e-04 -7.1615077e-06;
%  -1.0391317e-04  3.8396441e-04  2.5702146e-05;
%  -7.1615077e-06  2.5702146e-05  1.0527541e-04]



% kitti
R = [3.7570933e-06  1.1800409e-07 -8.0405171e-06;
  1.1800409e-07  9.9594472e-06 -4.1721264e-06;
 -8.0405171e-06 -4.1721264e-06  1.5025614e-04]

R = [10^-2 0 0;
    0 1 0;
    0 0 1]*10^-3
R = [2.5344568e-06 -3.3911829e-06  1.0133750e-06;
 -3.3911829e-06  1.2174246e-05 -5.3162039e-06;
  1.0133750e-06 -5.3162039e-06  7.5283851e-06]


acc_gnd = mylp(acc_gnd, 0.9);

for i=1:1:N
    velKF(i+1,:) = A*velKF(i,:)' + dt(i)*acc_gnd(i,:)';
    %R = acc_Q{i};
    pp = A*P{i}*A' + R;
    
    mCov = dtr_Q_gnd{i};
    K = pp*H'*inv(H*pp*H' + mCov);
    z = pr_dtr_gnd(i,:)';
    velKF(i+1,:) = (velKF(i+1,:)' + K*(z-H*velKF(i+1,:)'))';
    P{i+1} = pp - K*H*pp;
end

for i =1:1:N
   kfcov3(i,:) = diag(P{i});
end
kfstd3 = sqrt(kfcov3);

pos_intKF = cumtrapz(velKF(1:end-1,:));
pr_pos_int = cumtrapz(pr_dtr_gnd);

%% plot

w = 500;
h = 600;
fig = figure('Renderer', 'painters', 'Position', [600 100 w h]);
subplot(3,1, 1)
hold on
plot(gt_dtr_gnd(:,1),'r.-', 'MarkerSize',5);
plot(pr_dtr_gnd(:,1), 'b.-', 'MarkerSize',2);
plot(vel_imu(:,1),'cyan.-', 'MarkerSize',2);
plot(velKF(:,1), 'g.-', 'MarkerSize',2);
ylabel('X axis, m', 'fontsize', 14)
legend('GT', 'CNN', 'ACC', 'KF')

subplot(3,1, 2)
hold on
plot(gt_dtr_gnd(:,2),'r.-', 'MarkerSize',5);
plot(pr_dtr_gnd(:,2), 'b.-', 'MarkerSize',2);
plot(vel_imu(:,2),'cyan.-', 'MarkerSize',2);
plot(velKF(:,2), 'g.-', 'MarkerSize',2);
ylabel('Y axis, m', 'fontsize', 14)
legend('GT', 'CNN', 'ACC', 'KF')

subplot(3,1, 3)
hold on
plot(gt_dtr_gnd(:,3),'r.-', 'MarkerSize',5);
plot(pr_dtr_gnd(:,3), 'b.-', 'MarkerSize',2);
plot(vel_imu(:,3),'cyan.-', 'MarkerSize',2);
plot(velKF(:,3), 'g.-', 'MarkerSize',2);
ylabel('Z axis, m', 'fontsize', 16)
legend('GT', 'CNN', 'ACC', 'KF')
xlabel('Data Points', 'fontsize', 16)

w = 500;
h = 600;
fig = figure('Renderer', 'painters', 'Position', [600 100 w h]);
subplot(3,1, 1)
hold on
plot(kfstd3(:,1),'b.', 'MarkerSize',1);
ylabel('X axis, m', 'fontsize', 16)

subplot(3,1, 2)
hold on
plot(kfstd3(:,2),'b.', 'MarkerSize',1);
ylabel('Y axis, m', 'fontsize', 16)

subplot(3,1, 3)
hold on
plot(kfstd3(:,3),'b.', 'MarkerSize',1);
ylabel('Z axis, m', 'fontsize', 16)
xlabel('Data Points', 'fontsize', 16)

figure
subplot(3,1, 1)
hold on
plot(gt_pos(:,1),'r.-', 'MarkerSize',5);
plot(pr_pos(:,1),'b.-', 'MarkerSize',1);
plot(pos_intKF(:,1),'g.-', 'MarkerSize',1);
ylabel('X axis, m', 'fontsize', 16)
legend('GT', 'CNN', 'KF');
subplot(3,1, 2)
hold on
plot(gt_pos(:,2),'r.-', 'MarkerSize',5);
plot(pr_pos(:,2),'b.-', 'MarkerSize',1);
plot(pos_intKF(:,2),'g.-', 'MarkerSize',1);
ylabel('Y axis, m', 'fontsize', 16)
legend('GT', 'CNN', 'KF');
subplot(3,1, 3)
hold on
plot(gt_pos(:,3),'r.-', 'MarkerSize',5);
plot(pr_pos(:,3),'b.-', 'MarkerSize',1);
plot(pos_intKF(:,3),'g.-', 'MarkerSize',1);
ylabel('Z axis, m', 'fontsize', 16)
legend('GT', 'CNN', 'KF');
xlabel('Data Points', 'fontsize', 16)

figure
hold on
plot(gt_pos(:,1), gt_pos(:,3), 'r')
plot(pr_pos(:,1), pr_pos(:,3), 'b')
plot(pos_intKF(:,1), pos_intKF(:,3), 'g')
xlabel('Position X, m', 'fontsize', 16)
ylabel('Position Y, m', 'fontsize', 16)
legend('GT', 'CNN', 'KF');


prPath = ['Data\',getPRPath(dsName, subType, seq)];
kfName = strcat(prPath, '_dtrKF.txt')
dlmwrite(kfName, velKF(2:end,:))



function[pltIndex] = mysubplot(gt, pr, index)
    hold on
    grid on
    plot(gt, 'r.', 'MarkerSize',10)
    plot(pr, 'b.', 'MarkerSize',1)
    hold off
    pltIndex = index + 1;
end

function[pltIndex] = mysubplot2D(gt1, gt2, pr1, pr2, index)
    hold on
    grid on
    plot(gt1,gt2, 'r.', 'MarkerSize',10)
    plot(pr1,pr2, 'b.', 'MarkerSize',1)
    hold off
    pltIndex = index + 1;
end


function[res] = mylp(data, a)
    res(1,:) = data(1,:);
    for i =2:1:length(data)
        res(i,:) = a*res(i-1,:) + (1-a)*data(i,:);
    end
end

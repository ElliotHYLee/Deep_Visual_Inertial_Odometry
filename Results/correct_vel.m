clc, clear, close all
dsName = 'airsim';
subType = 'mr';
seq = 1;

%% Get Ground Truth Info.
gtPath = getGTPath(dsName,subType, seq);
gt_dtName = strcat(gtPath, 'dt.txt');
gt_duName = strcat(gtPath, '\du.txt');
gt_dwName = strcat(gtPath, '\dw.txt');
gt_dtrName = strcat(gtPath, '\dtrans.txt');
gt_dtr_gndName = strcat(gtPath, '\dtrans_gnd.txt');
linRName = strcat(gtPath, '\linR.txt');
gt_posName = strcat(gtPath, '\pos.txt');
acc_gndName = strcat(gtPath, '\acc_gnd.txt');

dt = importdata(gt_dtName);
gt_du = importdata(gt_duName);
gt_dw = importdata(gt_dwName);
gt_dtr  = importdata(gt_dtrName);
gt_dtr_gnd  = importdata(gt_dtr_gndName);
linR = importdata(linRName);
gt_pos = importdata(gt_posName);
gt_pos = gt_pos - gt_pos(1,:);
acc_gnd = importdata(acc_gndName);

%% Get Prediction Info.
prPath = ['Data\',getPRPath(dsName, subType, seq)];
pr_duName = strcat(prPath, '_du.txt');
pr_dwName = strcat(prPath, '_dw.txt');
pr_dtr_gndName = strcat(prPath, '_dtr_gnd.txt');
pr_duCovName = strcat(prPath, '_du_cov.txt');
pr_dwCovName = strcat(prPath, '_dw_cov.txt');
pr_dtrCovName = strcat(prPath, '_dtr_cov.txt');

pr_du = importdata(pr_duName);
pr_dw = importdata(pr_dwName);
pr_dtr_gnd = importdata(pr_dtr_gndName);
pr_du_cov = importdata(pr_duCovName);
pr_dw_cov = importdata(pr_dwCovName);
pr_dtr_cov = importdata(pr_dtrCovName);

N = length(pr_du);
[du_Q, du_cov3] = getCov(pr_du_cov);
[dw_Q, dw_cov3] = getCov(pr_dw_cov);
[dtr_Q, dtr_cov3] = getCov(pr_dtr_cov);
du_std3 = sqrt(du_cov3);
dw_std3 = sqrt(dw_cov3);
dtr_std3 = sqrt(dtr_cov3);

for i =1:1:N
   rotm = reshape(linR(i,:), 3,3)';
   dtr_Q_gnd{i} = rotm*dtr_Q{i}*rotm';
   cov3(i,:) = diag(dtr_Q_gnd{i});
end

dtr_gnd_std3 = sqrt(cov3);

time = cumtrapz(dt);
% if strcmp(dsName, 'kitti')
%    %acc_gnd = dt.*acc_gnd;
%     vel_imu = cumtrapz(time, acc_gnd);
% else
%     acc_gnd = dt.*acc_gnd;
% vel_imu = cumtrapz(time, acc_gnd);
% end
acc_gnd = dt.*acc_gnd;
vel_imu = cumtrapz(time, acc_gnd);

% KF
velKF = [0 0 0];
A = eye(3);
H = eye(3);
P{1} = eye(3)*10^-10;
R = [10^0 0 0; 0 1 0; 0 0 10^0]*10^-4
for i=1:1:N
    velKF(i+1,:) = A*velKF(i,:)' + 0.5*dt(i)*acc_gnd(i,:)';
    pp = A*P{i}*A' + R;
%     P{i+1} = pp;
    mCov = dtr_Q_gnd{i};
    %mCov = eye(3)*10^-1;
    if (mod(i, 100))
        K = pp*H'*inv(H*pp*H' + mCov)
        z = pr_dtr_gnd(i,:)';
        velKF(i+1,:) = (velKF(i,:)' + K*(z-H*velKF(i,:)'))';
        P{i+1} = pp - K*H*pp;
    else
        P{i+1} = pp;
    end
end


for i =1:1:N
   kfcov3(i,:) = diag(P{i});
end
kfstd3 = sqrt(kfcov3);

pos_intKF = cumtrapz(velKF(1:end-1,:));
pr_pos_int = cumtrapz(pr_dtr_gnd);

%% plot
subPlotRow = 8;
subPlotCol = 3;
index = 1;
w = 1000;
h = 1200;
fig = figure('Renderer', 'painters', 'Position', [600 100 w h]);

axes( 'Position', [0, 0.95, 1, 0.05] ) ;
set( gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None' ) ;
figTitle = [dsName, ' ', subType, ' ', int2str(seq)];
text( 0.5, 0, figTitle, 'FontSize', 20', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;

text( 0.75, 0, '- Ground Truth', 'FontSize', 10', 'Color', 'red', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
text( 0.737, -0.3, '- Predicted', 'FontSize', 10', 'Color', 'blue', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;

% plot du
for i=1:1:3
    subplot(subPlotRow, subPlotCol, index)
    index = mysubplot(gt_du(:,i), pr_du(:,i), index);
    if i==1 strYLabel = 'x';
    elseif i==2 strYLabel = 'y';
    else strYLabel = 'z';
    end
    ylabel([strcat('\fontsize{14} {\Delta}u_', strYLabel, ', m')])
    xlabel(['\fontsize{14} Data Points'])
end

% plot du_std
for i=1:1:3
    subplot(subPlotRow, subPlotCol, index)
    plot(du_std3(:,i), 'b.', 'MarkerSize',5);
    grid on
    index = index + 1;
    if i==1 strYLabel = 'x';
    elseif i==2 strYLabel = 'y';
    else strYLabel = 'z';
    end
    ylabel([strcat('\fontsize{14} 1{\sigma}_{{\Delta}u_', strYLabel, '}, m')])
    xlabel(['\fontsize{14} Data Points'])
end

% plot dw
for i=1:1:3
    subplot(subPlotRow, subPlotCol, index)
    index = mysubplot(gt_dw(:,i), pr_dw(:,i), index);
    if i==1 strYLabel = 'x';
    elseif i==2 strYLabel = 'y';
    else strYLabel = 'z';
    end
    ylabel([strcat('\fontsize{14} {\Delta}w_', strYLabel, ', rad')])
    xlabel(['\fontsize{14} Data Points'])
end

% plot dw_std
for i=1:1:3
    subplot(subPlotRow, subPlotCol, index)
    plot(dw_std3(:,i), 'b.', 'MarkerSize',5);
    grid on
    index = index + 1;
    if i==1 strYLabel = 'x';
    elseif i==2 strYLabel = 'y';
    else strYLabel = 'z';
    end
    ylabel([strcat('\fontsize{14} 1{\sigma}_{{\Delta}w_', strYLabel, '}, rad')])
    xlabel(['\fontsize{14} Data Points'])
end

% plot vel_gnd_imu
for i=1:1:3
    subplot(subPlotRow, subPlotCol, index)
    hold on
    plot(gt_dtr_gnd(:,i), 'r.', 'MarkerSize',5);
    plot(vel_imu(:,i), 'g.', 'MarkerSize',2);
    plot(pr_dtr_gnd(:,i), 'b.', 'MarkerSize',1);
    grid on
    title('Vel Gnd Imu in Green')
    index = index + 1;
    if i==1 strYLabel = 'x';
    elseif i==2 strYLabel = 'y';
    else strYLabel = 'z';
    end
    ylabel([strcat('\fontsize{14} Vel Gnd ', strYLabel, ', m/s')])
    xlabel(['\fontsize{14} Data Points'])
end

% plot vel_gnd_KF
for i=1:1:3
    subplot(subPlotRow, subPlotCol, index)
    hold on
    plot(gt_dtr_gnd(:,i), 'r.', 'MarkerSize',5);
    plot(velKF(:,i), 'g.', 'MarkerSize',2);
    grid on
    index = index + 1;
    if i==1 strYLabel = 'x';
    elseif i==2 strYLabel = 'y';
    else strYLabel = 'z';
    end
    ylabel([strcat('\fontsize{14} Vel Gnd_', strYLabel, ', m/s')])
    xlabel(['\fontsize{14} Data Points'])
end

% plot kf std
for i=1:1:3
    subplot(subPlotRow, subPlotCol, index)
    plot(kfstd3(:,i), 'b.', 'MarkerSize',5);
    grid on
    index = index + 1;
    if i==1 strYLabel = 'x';
    elseif i==2 strYLabel = 'y';
    else strYLabel = 'z';
    end
    ylabel([strcat('\fontsize{14} 1{\sigma}_{kf_', strYLabel, '}, rad')])
    xlabel(['\fontsize{14} Data Points'])
end

for i=1:1:3
    subplot(subPlotRow, subPlotCol, index)
    hold on
    plot(gt_pos(:,i), 'r.', 'MarkerSize',5);
    plot(pos_intKF(:,i), 'g.', 'MarkerSize',2);
    plot(pr_pos_int(:,i), 'b.', 'MarkerSize',1);
    grid on
    index = index + 1;
    if i==1 strYLabel = 'x';
    elseif i==2 strYLabel = 'y';
    else strYLabel = 'z';
    end
    ylabel([strcat('\fontsize{14} Vel Gnd_', strYLabel, ', m/s')])
    xlabel(['\fontsize{14} Data Points'])
end

figure
subplot(3,1, 1)
hold on
plot(gt_dtr_gnd(:,1),'r.', 'MarkerSize',5);
plot(pr_dtr_gnd(:,1), 'b.-', 'MarkerSize',2);
plot(vel_imu(:,1),'cyan.-', 'MarkerSize',2);
plot(velKF(:,1), 'g.', 'MarkerSize',2);
legend('gt', 'cnn', 'imu', 'kf')

subplot(3,1, 2)
hold on
plot(gt_dtr_gnd(:,2),'r.', 'MarkerSize',5);
plot(pr_dtr_gnd(:,2), 'b.-', 'MarkerSize',2);
plot(vel_imu(:,2),'cyan.-', 'MarkerSize',2);
plot(velKF(:,2), 'g.', 'MarkerSize',2);
legend('gt', 'cnn', 'imu', 'kf')

subplot(3,1, 3)
hold on
plot(gt_dtr_gnd(:,3),'r.', 'MarkerSize',5);
plot(pr_dtr_gnd(:,3), 'b.-', 'MarkerSize',2);
plot(vel_imu(:,3),'cyan.-', 'MarkerSize',2);
plot(velKF(:,3), 'g.', 'MarkerSize',2);
legend('gt', 'cnn', 'imu', 'kf')


figure
hold on
plot(gt_pos(:,1), gt_pos(:,2), 'ro')
plot(pos_intKF(:,1), pos_intKF(:,2), 'b.')




dlmwrite('../velKF.txt', velKF)


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


clc, clear, close all
dsName = 'airsim';
subType = 'mr';
seq = 2;

loadData

time = cumtrapz(dt);
acc_gnd = dt.*acc_gnd;
vel_imu = cumtrapz(time, acc_gnd);

% KF
velKF = [0 0 0];
A = eye(3);
H = eye(3);
P{1} = eye(3)*10^-10;
R = [1 0 0; 0 1 0; 0 0 1]*10^-4
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
plot(gt_pos(:,2), gt_pos(:,1), 'r')
plot(pr_pos(:,2), pr_pos(:,1), 'b')
plot(pos_intKF(:,2), pos_intKF(:,1), 'g')

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





















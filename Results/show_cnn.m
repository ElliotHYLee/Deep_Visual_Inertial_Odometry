clc, clear, close all
dsName = 'airsim';
subType = 'mrseg';
seq = 2;

%% Get Ground Truth Info.
gtPath = getGTPath(dsName,subType, seq);
gt_dtName = strcat(gtPath, 'dt.txt');
gt_duName = strcat(gtPath, '\du.txt');
gt_dwName = strcat(gtPath, '\dw.txt');
gt_dtrName = strcat(gtPath, '\dtrans.txt');
gt_dtr_gndName = strcat(gtPath, '\dtrans_gnd.txt');
linRName = strcat(gtPath, '\linR.txt');

dt = importdata(gt_dtName);
gt_du = importdata(gt_duName);
gt_dw = importdata(gt_dwName);
gt_dtr  = importdata(gt_dtrName);
gt_dtr_gnd  = importdata(gt_dtr_gndName);
linR = importdata(linRName);

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

%% Do se(3) -> SE(3)
lie = Lie();
se3 = LieSE3();
so3 = LieSO3();

gt_T{1} = eye(4);
pr_T{1} = eye(4);

for i = 1:1:N
    pr_dT = se3.getExp(gt_dw(i,:)', pr_du(i,:)');
    gt_dT = se3.getExp(gt_dw(i,:)', gt_du(i,:)');
    gt_T{i+1} = gt_T{i}*gt_dT;
    pr_T{i+1} = pr_T{i}*pr_dT;
    gt_pos(i,:) = gt_T{i}(1:3,4)';
    pr_pos(i,:) = pr_T{i}(1:3,4)';
end

%% plot
subPlotRow = 6;
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

% plot position
for i=1:1:3
    subplot(subPlotRow, subPlotCol, index)
    index = mysubplot(gt_pos(:,i), pr_pos(:,i), index);
    if i==1 strYLabel = 'x';
    elseif i==2 strYLabel = 'y';
    else strYLabel = 'z';
    end
    ylabel([strcat('\fontsize{14} Position_', strYLabel, ', m')])
    xlabel(['\fontsize{14} Data Points'])
end

% plot position 2D
subplot(subPlotRow, subPlotCol, index)
index = mysubplot2D(gt_pos(:,2), gt_pos(:,1), pr_pos(:,2), pr_pos(:,1), index);
ylabel(['\fontsize{14} Position_x, m'])
xlabel(['\fontsize{14} Position_y, m'])

subplot(subPlotRow, subPlotCol, index)
index = mysubplot2D(gt_pos(:,1), gt_pos(:,3), pr_pos(:,1), pr_pos(:,3), index);
ylabel(['\fontsize{14} Position_z, m'])
xlabel(['\fontsize{14} Position_x, m'])

subplot(subPlotRow, subPlotCol, index)
index = mysubplot2D(gt_pos(:,2), gt_pos(:,3), pr_pos(:,2), pr_pos(:,3), index);
ylabel(['\fontsize{14} Position_z, m'])
xlabel(['\fontsize{14} Position_x, m'])


err = abs(gt_du-pr_du);
mae = mean(err)
cov(err)

figName = strcat('Figures\',getPRPath(dsName, subType, seq), '_results.png');
saveas(fig, figName)

err_dtr_gnd = gt_dtr_gnd- pr_dtr_gnd;

figure
subplot(4,3,1)
plot(err_dtr_gnd(:,1))
grid on 
grid minor

subplot(4,3,2)
plot(err_dtr_gnd(:,2))
grid on 
grid minor

subplot(4,3,3)
plot(err_dtr_gnd(:,3))
grid on 
grid minor

for i =1:1:N
   rotm = reshape(linR(i,:), 3,3)';
   dtr_Q_gnd{i} = rotm*dtr_Q{i}*rotm';
   cov3(i,:) = diag(dtr_Q_gnd{i});
end

dtr_gnd_std3 = sqrt(cov3);
subplot(4,3,4)
plot(dtr_gnd_std3(:,1))
grid on 
grid minor

subplot(4,3,5)
plot(dtr_gnd_std3(:,2))
grid on 
grid minor

subplot(4,3,6)
plot(dtr_gnd_std3(:,3))
grid on 
grid minor

subplot(4,3,7)
hold on
plot(gt_dtr_gnd(:,1),'ro')
plot(pr_dtr_gnd(:,1), 'b.')
grid on 
grid minor

subplot(4,3,8)
hold on
plot(gt_dtr_gnd(:,2),'ro')
plot(pr_dtr_gnd(:,2), 'b.')
grid on 
grid minor

subplot(4,3,9)
hold on
plot(gt_dtr_gnd(:,3),'ro')
plot(pr_dtr_gnd(:,3), 'b.')
grid on 
grid minor




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











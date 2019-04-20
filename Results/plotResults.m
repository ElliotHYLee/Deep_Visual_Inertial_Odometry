clc, clear, close all
dsName = 'airsim';
subType = 'mrseg';
seq=0;

%% Get Ground Truth Info.
gtPath = getGTPath(dsName,subType, seq);
gt_dtName = strcat(gtPath, 'dt.txt');
gt_duName = strcat(gtPath, '\du.txt');
gt_dwName = strcat(gtPath, '\dw.txt');
gt_dtrName = strcat(gtPath, '\dtrans.txt');
gt_dtr_gndName = strcat(gtPath, '\dtrans_gnd.txt');
gt_pos_gndName = strcat(gtPath, '\pos.txt');
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
pr_dtrName = strcat(prPath, '_dtr.txt');
pr_dtr_gndName = strcat(prPath, '_dtr_gnd.txt');
pr_duCovName = strcat(prPath, '_du_cov.txt');
pr_dwCovName = strcat(prPath, '_dw_cov.txt');
pr_dtrCovName = strcat(prPath, '_dtr_cov.txt');

pr_du = importdata(pr_duName);
pr_dw = importdata(pr_dwName);
pr_dtr = importdata(pr_dtrName);
pr_dtr_gnd = importdata(pr_dtr_gndName);
pr_du_cov = importdata(pr_duCovName);
pr_dw_cov = importdata(pr_dwCovName);
pr_dtr_cov = importdata(pr_dtrCovName);


w = 500;
h = 600;
fig = figure('Renderer', 'painters', 'Position', [600 100 w h]);
axes('Position', [0, 0.95, 1, 0.05] ) ;
set( gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None' ) ;
text( 0.75, 0, '- Ground Truth', 'FontSize', 10', 'Color', 'red', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
text( 0.737, -0.3, '- Predicted', 'FontSize', 10', 'Color', 'blue', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
subplot(3,1,1)
plot(gt_du(:,1), 'r')
hold on
plot(pr_du(:,1), 'b.', 'markersize', 1)
ylabel('X axis, m', 'fontsize', 14)
ylim([-0.1, 2])
subplot(3,1,2)
plot(gt_du(:,2), 'r')
hold on
plot(pr_du(:,2), 'b.', 'markersize', 1)
ylabel('Y axis, m', 'fontsize', 14)
subplot(3,1,3)
plot(gt_du(:,3), 'r')
hold on
plot(pr_du(:,3), 'b.', 'markersize', 1)
ylabel('Z axis, m', 'fontsize', 14)



w = 500;
h = 600;
fig = figure('Renderer', 'painters', 'Position', [600 100 w h]);
axes('Position', [0, 0.95, 1, 0.05] ) ;
set( gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None' ) ;
text( 0.75, 0, '- Ground Truth', 'FontSize', 10', 'Color', 'red', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
text( 0.737, -0.3, '- Predicted', 'FontSize', 10', 'Color', 'blue', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
subplot(3,1,1)
plot(gt_dw(:,1), 'r')
hold on
plot(pr_dw(:,1), 'b.', 'markersize', 1)
ylabel('X axis, m', 'fontsize', 14)
subplot(3,1,2)
plot(gt_dw(:,2), 'r')
hold on
plot(pr_dw(:,2), 'b.', 'markersize', 1)
ylabel('Y axis, m', 'fontsize', 14)
subplot(3,1,3)
plot(gt_dw(:,3), 'r')
hold on
plot(pr_dw(:,3), 'b.', 'markersize', 1)
ylabel('Z axis, m', 'fontsize', 14)












%% Do se(3) -> SE(3)
lie = Lie();
se3 = LieSE3();
so3 = LieSO3();

gt_T{1} = eye(4);
pr_T{1} = eye(4);
N = length(pr_du);
for i = 1:1:N
    pr_dT = se3.getExp(gt_dw(i,:)', pr_du(i,:)');
    gt_dT = se3.getExp(gt_dw(i,:)', gt_du(i,:)');
    gt_T{i+1} = gt_T{i}*gt_dT;
    pr_T{i+1} = pr_T{i}*pr_dT;
    gt_pos(i,:) = gt_T{i}(1:3,4)';
    pr_pos(i,:) = pr_T{i}(1:3,4)';
end

figure
plot(gt_pos(:,2), gt_pos(:,1), 'r', 'displayname', 'Ground Truth')
hold on
plot(pr_pos(:,2), pr_pos(:,1), 'b', 'markersize', 1, 'displayname', 'Predicted')
legend
ylabel('Y Position, m', 'fontsize', 14)
xlabel('X Position, m', 'fontsize', 14)






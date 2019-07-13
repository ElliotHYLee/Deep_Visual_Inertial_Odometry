clc, clear, close all
dsName = 'mycar';
subType = 'none';
seq = 1;

%% Get Ground Truth Info.
gtPath = getGTPath(dsName,subType, seq)
gt_dtName = strcat(gtPath, 'dt.txt')
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
gt_pos = importdata(gt_pos_gndName);
linR = importdata(linRName);

N = length(gt_du);

figure
plot3(gt_pos(:,1), gt_pos(:,2), gt_pos(:,3), 'r')
grid on
view(45,30)
title('Position')
xlabel('Position_X, m', 'fontsize', 16)
ylabel('Position_Y, m', 'fontsize', 16)
zlabel('Position_Z, m', 'fontsize', 16)
% zlim([-100 100])



w = 1000;
h = 1200;
fig = figure('Renderer', 'painters', 'Position', [600 100 w h]);
axes('Position', [0, 0.95, 1, 0.05] ) ;
set( gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None' ) ;
strTitle = strcat(dsName, {' '}, subType, {' '}, 'Sequence', {' '}, int2str(seq));
text( 0.5, 0, strTitle, 'FontSize', 16', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;

subplot(6,2,[1,3,5])
plot3(gt_pos(:,1), gt_pos(:,2), gt_pos(:,3), 'r')
grid on
view(45,30)
title('Position')
xlabel('Position_X, m', 'fontsize', 16)
ylabel('Position_Y, m', 'fontsize', 16)
zlabel('Position_Z, m', 'fontsize', 16)
zlim([-100 100])

subplot(6,2,2)
plot(gt_du(:,1), 'r.')
% ylim([-0.1, 2])
ylabel('dU_X, m', 'fontsize', 16)
xlabel('Data Points', 'fontsize', 16)
title('Tranlational Movement')
subplot(6,2,4)
plot(gt_du(:,2), 'r.')
ylabel('dU_Y, m', 'fontsize', 16)
xlabel('Data Points', 'fontsize', 16)
subplot(6,2,6)
plot(gt_du(:,3), 'r.')
ylabel('dU_Z, m', 'fontsize', 16)
xlabel('Data Points', 'fontsize', 16)

subplot(6,2,7)
plot(gt_dw(:,1), 'r.')
ylabel('dW_X, rad', 'fontsize', 16)
xlabel('Data Points', 'fontsize', 16)
title('Angular Displacement')
subplot(6,2,9)
plot(gt_dw(:,2), 'r.')
ylabel('dW_Y, rad', 'fontsize', 16)
xlabel('Data Points', 'fontsize', 16)
subplot(6,2,11)
plot(gt_dw(:,3), 'r.')
ylabel('dW_Z, rad', 'fontsize', 16)
xlabel('Data Points', 'fontsize', 16)

 
subplot(6,2,8)
plot(gt_dtr(:,1), 'r.')
ylabel('dtr_X, rad', 'fontsize', 16)
xlabel('Data Points', 'fontsize', 16)
% ylim([-0.1, 2])
title('Overall Displacement')
subplot(6,2,10)
plot(gt_dtr(:,2), 'r.')
ylabel('dtr_Y, rad', 'fontsize', 16)
xlabel('Data Points', 'fontsize', 16)
subplot(6,2,12)
plot(gt_dtr(:,3), 'r.')
ylabel('dtr_Z, rad', 'fontsize', 16)
xlabel('Data Points', 'fontsize', 16)

















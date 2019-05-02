clc, clear, close all
dsName = 'airsim';
subType = 'mr';
seq = 0;



prName = strcat('Data/', getBranchName(), '_', dsName, '_', subType, '_' , int2str(seq), '_kfRES.txt');
pr = importdata(prName);
prPosName = strcat('Data/', getBranchName(), '_', dsName, '_', subType, '_' , int2str(seq), '_kfRESPos.txt');
prPos = importdata(prPosName);
gtName = strcat('Data/', getBranchName(), '_', dsName, '_', subType, '_' , int2str(seq), '_gtSignal.txt');
gt = importdata(gtName);
gtPosName = strcat('Data/', getBranchName(), '_', dsName, '_', subType, '_' , int2str(seq), '_gtSignalPos.txt');
gtPos = importdata(gtPosName);
covName = strcat('Data/', getBranchName(), '_', dsName, '_', subType, '_' , int2str(seq), '_sysCov.txt');
cov = importdata(covName);
std = sqrt(cov);
stdPos = cumsum(std);
N = length(pr);

w = 500;
h = 600;
fig = figure('Renderer', 'painters', 'Position', [600 100 w h]);
axes('Position', [0, 0.95, 1, 0.05] ) ;
set( gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None' ) ;
text( 0.75, 0, '- Ground Truth', 'FontSize', 10', 'Color', 'red', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
text( 0.737, -0.3, '- Predicted', 'FontSize', 10', 'Color', 'blue', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
subplot(3,1,1)
plot(gt(:,1), 'ro', 'markersize', 1)
hold on
plot(pr(:,1), 'b.', 'markersize', 1)
xlim([0 N])
ylabel('X axis, m', 'fontsize', 14)
% ylim([-0.1, 2])
subplot(3,1,2)
plot(gt(:,2), 'ro', 'markersize', 1)
hold on
plot(pr(:,2), 'b.', 'markersize', 1)
xlim([0 N])
ylabel('Y axis, m', 'fontsize', 14)
subplot(3,1,3)
plot(gt(:,3), 'ro', 'markersize', 1)
hold on
plot(pr(:,3), 'b.', 'markersize', 1)
xlim([0 N])
ylabel('Z axis, m', 'fontsize', 14)



w = 500;
h = 600;
fig = figure('Renderer', 'painters', 'Position', [600 100 w h]);
axes('Position', [0, 0.95, 1, 0.05] ) ;
set( gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None' ) ;
text( 0.75, 0, '- Ground Truth', 'FontSize', 10', 'Color', 'red', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
text( 0.737, -0.3, '- Predicted', 'FontSize', 10', 'Color', 'blue', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
subplot(3,1,1)
plot(std(:,1), 'b.', 'markersize', 1)
xlim([0 N])
ylabel('X axis, m', 'fontsize', 14)
subplot(3,1,2)
plot(std(:,2), 'b.', 'markersize', 1)
xlim([0 N])
ylabel('Y axis, m', 'fontsize', 14)
subplot(3,1,3)
plot(std(:,3), 'b.', 'markersize', 1)
xlim([0 N])
ylabel('Z axis, m', 'fontsize', 14)


figure
plot(gtPos(:,2), gtPos(:,1), 'r', 'displayname', 'Ground Truth')
hold on
plot(prPos(:,2), prPos(:,1), 'b', 'markersize', 1, 'displayname', 'Predicted')
legend
ylabel('Y Position, m', 'fontsize', 14)
xlabel('X Position, m', 'fontsize', 14)









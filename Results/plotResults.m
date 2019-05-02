% clc, clear, close all
% dsName = 'euroc';
% subType = 'none';
% seq=1;

loadData;


w = 500;
h = 600;
fig = figure('Renderer', 'painters', 'Position', [600 100 w h]);
axes('Position', [0, 0.95, 1, 0.05] ) ;
set( gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None' ) ;
text( 0.75, 0, '- Ground Truth', 'FontSize', 10', 'Color', 'red', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
text( 0.737, -0.3, '- Predicted', 'FontSize', 10', 'Color', 'blue', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
subplot(3,1,1)
plot(gt_dtr_gnd(:,1), 'r')
hold on
plot(pr_dtr_gnd(:,1), 'bo', 'markersize', 1)
xlim([0 N])
ylabel('X axis, m', 'fontsize', 14)
% ylim([-0.1, 2])
subplot(3,1,2)
plot(gt_dtr_gnd(:,2), 'r')
hold on
plot(pr_dtr_gnd(:,2), 'b.', 'markersize', 1)
xlim([0 N])
ylabel('Y axis, m', 'fontsize', 14)
subplot(3,1,3)
plot(gt_dtr_gnd(:,3), 'r')
hold on
plot(pr_dtr_gnd(:,3), 'b.', 'markersize', 1)
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
plot(dtr_gnd_std3(:,1), 'b.', 'markersize', 1)
xlim([0 N])
ylabel('X axis, m', 'fontsize', 14)
subplot(3,1,2)
plot(dtr_gnd_std3(:,2), 'b.', 'markersize', 1)
xlim([0 N])
ylabel('Y axis, m', 'fontsize', 14)
subplot(3,1,3)
plot(dtr_gnd_std3(:,3), 'b.', 'markersize', 1)
xlim([0 N])
ylabel('Z axis, m', 'fontsize', 14)


figure
plot(gt_pos(:,1), gt_pos(:,3), 'r', 'displayname', 'Ground Truth')
hold on
plot(pr_pos(:,1), pr_pos(:,3), 'b', 'markersize', 1, 'displayname', 'Predicted')
legend
ylabel('Y Position, m', 'fontsize', 14)
xlabel('X Position, m', 'fontsize', 14)






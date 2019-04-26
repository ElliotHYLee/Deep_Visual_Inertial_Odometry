clc, clear, close all
dsName = 'airsim';
subType = 'mrseg';
seq=2;

loadData

%% plot
subPlotRow = 6;
subPlotCol = 3;
index = 1;
w = 1000;
h = 1200;
fig = figure('Renderer', 'painters', 'Position', [600 100 w h]);

axes( 'Position', [0, 0.95, 1, 0.05] );
set( gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None' ) ;
figTitle = [dsName, ' ', subType, ' ', int2str(seq)];
% text( 0.5, 0, figTitle, 'FontSize', 20', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;

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
%     ylim([0.4, 0.6])
    grid on
    index = index + 1;
    if i==1 strYLabel = 'x';
    elseif i==2 strYLabel = 'y';
    else strYLabel = 'z';
    end
    ylabel([strcat('\fontsize{14} 1{\sigma}_{{\Delta}w_', strYLabel, '}, rad')])
    xlabel(['\fontsize{14} Data Points'])
end

% plot dtr_gnd
for i=1:1:3
    subplot(subPlotRow, subPlotCol, index)
    index = mysubplot(gt_dtr_gnd(:,i), pr_dtr_gnd(:,i), index);
    grid on

    if i==1 strYLabel = 'x';
    elseif i==2 strYLabel = 'y';
    else strYLabel = 'z';
    end
    ylabel([strcat('\fontsize{14} {{\Delta}dtr ground_', strYLabel, '}, m')])
    xlabel(['\fontsize{14} Data Points'])
end

% plot dtr_gnd_std
for i=1:1:3
    subplot(subPlotRow, subPlotCol, index)
    plot(dtr_gnd_std3(:,i), 'b.', 'MarkerSize',5);
    grid on
    index = index + 1;
    if i==1 strYLabel = 'x';
    elseif i==2 strYLabel = 'y';
    else strYLabel = 'z';
    end
    ylabel([strcat('\fontsize{14} 1{\sigma}_{{\Delta}dtr ground_', strYLabel, '}, m')])
    xlabel(['\fontsize{14} Data Points'])
end


figName = strcat('Figures\', getPRPath(dsName, subType, seq), '_results.png');
saveas(fig, figName)

x = 1:1:N;

figure
subplot(3,1,1)
hold on
plot(gt_pos(:,1),'r', 'DisplayName', 'Ground Truth')
plot(pr_pos(:,1), 'b', 'DisplayName', 'Predicted')
xlim([0, N])
ylabel(['\fontsize{14} Position_x, m'])
grid on 
grid minor
legend('Location', 'best')

subplot(3,1,2)
hold on
plot(gt_pos(:,2),'r', 'DisplayName', 'Ground Truth')
plot(pr_pos(:,2), 'b', 'DisplayName', 'Predicted')
xlim([0, N])
ylabel(['\fontsize{14} Position_y, m'])
grid on 
grid minor
legend('Location', 'best')

subplot(3,1,3)
hold on
plot(gt_pos(:,3),'r', 'DisplayName', 'Ground Truth')
plot(pr_pos(:,3), 'b', 'DisplayName', 'Predicted')
xlim([0, N])
ylabel(['\fontsize{14} Position_z, m'])
grid on 
grid minor
legend('Location', 'best')

figure
plot(gt_pos(:,2), gt_pos(:,1), 'r', 'DisplayName', 'Ground Truth')
hold on
plot(pr_pos(:,2), pr_pos(:,1), 'b', 'DisplayName', 'Predicted');
ylabel(['\fontsize{14} Position_x, m'])
xlabel(['\fontsize{14} Position_y, m'])
legend('Location', 'best')



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











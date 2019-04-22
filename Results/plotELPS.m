function[] = plotELPS(dsName, gt_pos, pr_pos, P)
    if strcmp(dsName,'kitti') 
        N = length(gt_pos);
        drawELPS(gt_pos, pr_pos, P, 500, 1, 3, 10, 3);
    elseif strcmp(dsName,'euroc')
        N = length(gt_pos);
        drawELPS(gt_pos, pr_pos, P, 100, 1, 2, 10, 3);
    elseif strcmp(dsName,'mycar')
        N = length(gt_pos);
        drawELPS(gt_pos, pr_pos, P, 500, 1, 2, 50, 3);
    elseif strcmp(dsName,'airsim')
        N = length(gt_pos);
        drawELPS(gt_pos, pr_pos, P, 1000, 2, 1, 10, 3);
    else
        N = length(gt_pos);
        drawELPS(gt_pos, pr_pos, P, N, 2, 1, 10, 3);
    end
end

function[] = drawELPS(gt_pos, pr_pos, P, lastPoint, ax1, ax2, skip, scale)
    figure
    a = plot(gt_pos(1:end,ax1), gt_pos(1:end,ax2), 'r')
    hold on
    b = plot(pr_pos(1:end,ax1), pr_pos(1:end,ax2),'b')
    
    
    for i=1:skip:lastPoint
       Q = P{i};
       [x,y] = getELPS(Q([ax1,ax2],[ax1,ax2]), scale);
       x = x + pr_pos(i,ax1);
       y = y + pr_pos(i,ax2);
       plot(x,y, 'g')
    end
    legend([a, b], {'Ground Truth', 'Prediction'})
    xlabel('Position_X, m', 'fontsize', 14)
    ylabel('Position_Y, m', 'fontsize', 14)
    axis equal
end
function[] = plotELPS(dsName, gt_pos, pr_pos, P)
    if strcmp(dsName,'kitti') 
        N = length(gt_pos);
        drawELPS(gt_pos, pr_pos, P, N, 1, 3, 100, 1);
    elseif strcmp(dsName,'euroc')
        N = length(gt_pos);
        drawELPS(gt_pos, pr_pos, P, N, 1, 2, 100, 1);
    else
        N = length(gt_pos);
        drawELPS(gt_pos, pr_pos, P, N, 1, 2, 100, 1);
    end
end


function[] = drawELPS(gt_pos, pr_pos, P, lastPoint, ax1, ax2, skip, scale)
    figure
    plot(gt_pos(1:lastPoint,ax1), gt_pos(1:lastPoint,ax2), 'r')
    hold on
    plot(pr_pos(1:lastPoint,ax1), pr_pos(1:lastPoint,ax2),'g')
    for i=1:skip:lastPoint
       Q = P{i};
       [x,y] = getELPS(Q([ax1,ax2],[ax1,ax2]), scale);
       x = x + pr_pos(i,ax1);
       y = y + pr_pos(i,ax2);
       plot(x,y, 'b')
    end
    axis equal
end
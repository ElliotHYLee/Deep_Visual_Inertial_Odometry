clc, clear, close all
dsName = 'mycar';
subType = 'none';
seq=1;


plotResults;


for i =1:1:N
   rotm = reshape(linR(i,:), 3,3)';
   dtr_Q_gnd{i} = rotm*dtr_Q{i}*rotm';
   cov3(i,:) = diag(dtr_Q_gnd{i});
end
dtr_gnd_std3 = sqrt(cov3);

%%
pos = [0 0 0];
A = eye(3);
H = eye(3);
P{1} = eye(3)*10^-5;
for i=1:1:N
    pos(i+1,:) = pos(i,:) + pr_dtr_gnd(i,:);
    P{i+1} = A*P{i}*A' + dtr_Q_gnd{i};%dtr_Q_gnd{i};
    xx = dtr_Q_gnd{i};
    [vec, val] = eig(xx);
end

plotELPS(dsName, gt_pos, pr_pos, P)

figure
subplot(3,1,1)
plot(gt_pos(:,1), 'r')
xlim([0 N])
hold on
plot(pr_pos(:,1), 'b')
for i =1:100:N
    std3 = P{i};
    std3 = sqrt(diag(std3));
    plot([i, i], [pr_pos(i,1)  pr_pos(i,1) + 3*std3(1)], 'g', 'linewidth', 1)
    plot([i, i], [pr_pos(i,1)  pr_pos(i,1) - 3*std3(1)], 'g', 'linewidth', 1)
end
ylabel('Position_X, m', 'fontsize', 14)
grid on
grid minor

subplot(3,1,2)
plot(gt_pos(:,2), 'r')
xlim([0 N])
hold on
plot(pr_pos(:,2), 'b')
for i =1:100:N
    std3 = P{i};
    std3 = sqrt(diag(std3));
    plot([i, i], [pr_pos(i,2)  pr_pos(i,2) + 3*std3(2)], 'g', 'linewidth', 1)
    plot([i, i], [pr_pos(i,2)  pr_pos(i,2) - 3*std3(2)], 'g', 'linewidth', 1)
end

ylabel('Position_Y, m', 'fontsize', 14)
grid on
grid minor


subplot(3,1,3)
plot(gt_pos(:,3), 'r')
xlim([0 N])
hold on
plot(pr_pos(:,3), 'b')
for i =1:100:N
    std3 = P{i};
    std3 = sqrt(diag(std3));
    plot([i, i], [pr_pos(i,3)  pr_pos(i,3) + 3*std3(3)], 'g', 'linewidth', 1)
    plot([i, i], [pr_pos(i,3)  pr_pos(i,3) - 3*std3(3)], 'g', 'linewidth', 1)
end
ylabel('Position_Z, m', 'fontsize', 14)
grid on
grid minor










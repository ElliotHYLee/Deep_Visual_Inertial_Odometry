clc, clear, close all

dsName = 'kitti';
subType = 'none';
seq = 5;
noises=[0 2 16];

posPR = {};
for ii =1:1:3
    noise = noises(ii)
    loadData;
    posGT = gt_pos;
    posPR{ii,1} = pr_pos;
    
end

figure
hold on
plot(posGT(:,1), posGT(:,3), 'r')
plot(posPR{1}(:,1), posPR{1}(:,3), 'g')
plot(posPR{2}(:,1), posPR{2}(:,3), 'b')
plot(posPR{3}(:,1), posPR{3}(:,3), 'k.')
xlim([-300 400])
ylim([-100 500])
lgd = legend('GT', '0 %', '2 %', '16 %')
lgd.FontSize = 20;
xlabel(['\fontsize{24} Position_x, m'])
ylabel(['\fontsize{24} Position_z, m'])

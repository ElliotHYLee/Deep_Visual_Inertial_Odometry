clc, clear, close all
dsName = 'kitti';
subType = 'none';
seq = 5;

%% Get Ground Truth Info.
gtPath = getGTPath(dsName,subType, seq);
gt_dtName = strcat(gtPath, 'dt.txt');
gt_duName = strcat(gtPath, '\du.txt');
gt_dwName = strcat(gtPath, '\dw.txt');
gt_dtrName = strcat(gtPath, '\dtrans.txt');
gt_dtr_gndName = strcat(gtPath, '\dtrans_gnd.txt');
linRName = strcat(gtPath, '\linR.txt');
gt_posName = strcat(gtPath, '\pos.txt');

dt = importdata(gt_dtName);
gt_du = importdata(gt_duName);
gt_dw = importdata(gt_dwName);
gt_dtr  = importdata(gt_dtrName);
gt_dtr_gnd  = importdata(gt_dtr_gndName);
linR = importdata(linRName);
gt_pos = importdata(gt_posName);
gt_pos = gt_pos - gt_pos(1,:);
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
R = eye(3)*10
for i=1:1:N
    pos(i+1,:) = pos(i,:) + pr_dtr_gnd(i,:);
    P{i+1} = A*P{i}*A' + dtr_Q_gnd{i};

%     pos(i+1,:) = A*pos(i,:)';
%     pp = A*P{i}*A' + R;
%     mCov = dtr_Q_gnd{i}
%     K = pp*H'*inv(H*pp*H'+mCov)
%     z = pos(i,:)' + pr_dtr_gnd(i,:)';
%     pos(i+1,:) = (pos(i,:)' + K*(z-H*pos(i,:)'))';
%     P{i+1} = pp - K*H*pp;
end

pos_int = cumtrapz(pr_dtr_gnd);

if strcmp(dsName,'kitti') 
    f = N
    figure
    plot(gt_pos(1:f,1), gt_pos(1:f,3), 'r')
    hold on
    plot(pos(1:f,1), pos(1:f,3),'b')
    plot(pos_int(1:f,1), pos_int(1:f,3),'g')
    for i=1:100:f
       Q = P{i};
       [x,y] = getELPS(Q([1,3],[1,3]), 9);
       x = x + pos(i,1);
       y = y + pos(i,3);
       plot(x,y, 'b')
    end
    
elseif strcmp(dsName,'euroc')
    f = 1500
    figure
    plot(gt_pos(1:f,2), gt_pos(1:f,1), 'r')
    hold on
    plot(pos(1:f,2), pos(1:f,1),'b')
    plot(pos_int(1:f,2), pos_int(1:f,1),'g')
    for i=1:50:f
       [x,y] = getELPS(Q([1,2],[1,2]), 1);
       x = x + pos(i,1);
       y = y + pos(i,2);
       plot(y,x, 'b')
    end
    
else
    f = N
    figure
    plot(gt_pos(1:f,2), gt_pos(1:f,1), 'r')
    hold on
    plot(pos(1:f,2), pos(1:f,1),'b')
    plot(pos_int(1:f,2), pos_int(1:f,1),'g')
    for i=1:50:f
        Q = P{i};
       [x,y] = getELPS(Q([1,2],[1,2]), 0.1);
       x = x + pos(i,1);
       y = y + pos(i,2);
       plot(y,x, 'b')
    end
    
end
axis equal



for i=1:1:N
    posCov3(i,:) = diag(P{i});
end

dPos_std = sqrt(posCov3);

% figure
% hold on
% plot(dPos_std(:,1), 'r.')
% plot(dPos_std(:,2), 'g.')
% plot(dPos_std(:,3), 'b.')
% % ylim([0 1])
% 
% figure
% hold on
% plot(dtr_gnd_std3(:,1), 'r.')
% plot(dtr_gnd_std3(:,2), 'g.')
% plot(dtr_gnd_std3(:,3), 'b.')
% % ylim([0 1])

function[x,y] = getELPS(Q, s)
[vec, val] = eig(Q);
val = diag(val);
[v, idx] = max(val);
V = vec(:,idx);
angle = atan(V(2)/V(1));
yaw = angle;
yaw*180/pi

a = 2*sqrt(val(1)*s);
b = 2*sqrt(val(2)*s);

amp = max([a,b]);


t = linspace(0,2*pi,1000);
theta0 = yaw;
x = amp*sin(t+theta0);
y = amp*cos(t);
end


















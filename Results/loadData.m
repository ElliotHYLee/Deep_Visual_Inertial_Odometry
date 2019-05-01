% clc, clear, close all
% dsName = 'airsim';
% subType = 'mrseg';
% seq = 0;

%% Get Ground Truth Info.
gtPath = getGTPath(dsName,subType, seq);
gt_dtName = strcat(gtPath, 'dt.txt');
gt_duName = strcat(gtPath, '\du.txt');
gt_dwName = strcat(gtPath, '\dw.txt');
gt_dwGyroName = strcat(gtPath, '\dw_gyro.txt');
gt_dtrName = strcat(gtPath, '\dtrans.txt');
gt_dtr_gndName = strcat(gtPath, '\dtrans_gnd.txt');
gt_pos_gndName = strcat(gtPath, '\pos.txt');
linRName = strcat(gtPath, '\linR.txt');
acc_gndName = strcat(gtPath, '\acc_gnd.txt');
acc_gnd = importdata(acc_gndName);


dt = importdata(gt_dtName);
gt_du = importdata(gt_duName);
gt_dw = importdata(gt_dwName);
gt_dw_gyro = importdata(gt_dwGyroName);
gt_dtr  = importdata(gt_dtrName);
gt_dtr_gnd  = importdata(gt_dtr_gndName);

linR = importdata(linRName);
gt_pos = importdata(gt_pos_gndName);
gt_pos = gt_pos - gt_pos(1,:);


%% Get Prediction Info.
prPath = ['Data\',getPRPath(dsName, subType, seq)];
pr_duName = strcat(prPath, '_du.txt');
pr_dwName = strcat(prPath, '_dw.txt');
pr_dtr_gndName = strcat(prPath, '_dtr_gnd.txt');
pr_dtrName = strcat(prPath, '_dtr.txt');
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


pr_pos = cumtrapz(pr_dtr_gnd);












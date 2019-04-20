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



















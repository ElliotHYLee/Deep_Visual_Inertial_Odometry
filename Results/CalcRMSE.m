clc, clear, close all
dsName = 'mycar';
subType = 'none';

for seq = 0:1:2

%% Get Ground Truth Info.
gtPath = getGTPath(dsName,subType, seq);
gt_dtName = strcat(gtPath, 'dt.txt');
gt_duName = strcat(gtPath, '\du.txt');
gt_dwName = strcat(gtPath, '\dw.txt');
gt_dtrName = strcat(gtPath, '\dtrans.txt');
gt_dtr_gndName = strcat(gtPath, '\dtrans_gnd.txt');
gt_pos_gndName = strcat(gtPath, '\pos.txt');
linRName = strcat(gtPath, '\linR.txt');

dt = importdata(gt_dtName);
gt_du = importdata(gt_duName);
gt_dw = importdata(gt_dwName);
gt_dtr  = importdata(gt_dtrName);
gt_dtr_gnd  = importdata(gt_dtr_gndName);
linR = importdata(linRName);

%% Get Prediction Info.
prPath = ['Data\',getPRPath(dsName, subType, seq)];
pr_duName = strcat(prPath, '_du.txt');
pr_dwName = strcat(prPath, '_dw.txt');
pr_dtrName = strcat(prPath, '_dtr.txt');
pr_dtr_gndName = strcat(prPath, '_dtr_gnd.txt');
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

duRMSE(seq+1, 1) = getRMSE(gt_du - pr_du);
dwRMSE(seq+1, 1) = getRMSE(gt_dw - pr_dw);
dtrRMSE(seq+1, 1) = getRMSE(pr_dtr - gt_dtr);


%% Velocity RMSE;
duRMSE;
dwRMSE;
dtrRMSE;


%% Position: Do se(3) -> SE(3)
lie = Lie();
se3 = LieSE3();
so3 = LieSO3();

gt_T{1} = eye(4);
pr_T{1} = eye(4);
N = length(pr_du);
for i = 1:1:N
    pr_dT = se3.getExp(gt_dw(i,:)', pr_du(i,:)');
    gt_dT = se3.getExp(gt_dw(i,:)', gt_du(i,:)');
    gt_T{i+1} = gt_T{i}*gt_dT;
    pr_T{i+1} = pr_T{i}*pr_dT;
    gt_pos(i,:) = gt_T{i}(1:3,4)';
    pr_pos(i,:) = pr_T{i}(1:3,4)';
end

%% Position RMSE
posRMSE(seq+1, 1) = getRMSE(gt_pos - pr_pos);

end

vel_RMSE = [duRMSE, dwRMSE, dtrRMSE]
posRMSE

function[result] = getRMSE3(err)
    N = length(err);
    se = err.^2;
    sse = sum(se);
    mse = sse/N;
    rmse = sqrt(mse);    
    result = rmse;
end

function[result] = getRMSE(err)
    rmse3 = getRMSE3(err);
    result = mean(rmse3);
end






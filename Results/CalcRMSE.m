clc, clear, close all
dsName = 'euroc';
subType = 'none';

for seq = 1:1:5
    loadData;
    duRMSE(seq+1, 1) = getRMSE(gt_du - pr_du);
    dwRMSE(seq+1, 1) = getRMSE(gt_dw - pr_dw);
    dtrRMSE(seq+1, 1) = getRMSE(pr_dtr - gt_dtr);
    dtr_gndRMSE(seq+1, 1) = getRMSE(pr_dtr_gnd - gt_dtr_gnd);
    velKFRMSE(seq+1, 1) = getRMSE(velKF - gt_dtr_gnd);
    %% Velocity RMSE;
    duRMSE;
    dwRMSE;
    dtrRMSE;

    %% Position RMSE
    posRMSE(seq+1, 1) = getRMSE(gt_pos - pr_pos);
    KFposRMSE(seq+1, 1) = getRMSE(gt_pos - posKF);

    %% Position RMSE per 100 iteration
    seqRMSE(seq+1,:) = getRMSE100(pr_pos, gt_pos, N, seq)
    kfseqRMSE(seq+1,:) = getRMSE100(posKF, gt_pos, N, seq)
end

allRMSE = [duRMSE, dwRMSE, dtrRMSE, dtr_gndRMSE, posRMSE, seqRMSE]
acorrRMSE = [velKFRMSE, KFposRMSE, kfseqRMSE]


function[result] = getRMSE100(pr_pos, gt_pos, N, seq)
    idx = 1:100:N;
    sumSeqRMSE = 0;
    for i =2:1:length(idx)
        s = idx(i-1);
        f = idx(i);
        seq_pr = pr_pos(s:f,:);
        seq_pr = seq_pr - seq_pr(1,:);
        seq_gt = gt_pos(s:f,:);
        seq_gt = seq_gt - seq_gt(1,:);
        err = (seq_pr - seq_gt);
        sumSeqRMSE = sumSeqRMSE + getRMSE(err);
    end
    result = sumSeqRMSE/length(idx);
end

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






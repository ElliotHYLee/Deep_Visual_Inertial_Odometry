clc, clear, close all
dsName = 'kitti';
subType = 'none';
noise = 16;

for seq = 5:1:5

    loadData;
    %% Position RMSE
    posRMSE(seq+1, 1) = getRMSE(gt_pos - pr_pos);

    %% Position RMSE per 100 iteration
    N;
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
    seqRMSE(seq+1,:) = sumSeqRMSE/length(idx);

end

% allRMSE = [duRMSE, dwRMSE, dtrRMSE, posRMSE, seqRMSE]
% 
% 
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






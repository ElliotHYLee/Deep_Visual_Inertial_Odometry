clc, clear, close all
dsName = 'kitti';
subType = 'none';
noise=0;
for seq = [0 1 2 3 4 5 6 7 8 9 10]
    loadData;
%     duRMSE(seq+1, 1) = getRMSE(gt_du - pr_du);
%     dwRMSE(seq+1, 1) = getRMSE(gt_dw - pr_dw);
%     dtrRMSE(seq+1, 1) = getRMSE(pr_dtr - gt_dtr);
%     dtr_gndRMSE(seq+1, 1) = getRMSE(pr_dtr_gnd - gt_dtr_gnd);
%     velKFRMSE(seq+1, 1) = getRMSE(velKF - gt_dtr_gnd);
    %% Velocity RMSE;
%     duRMSE;
%     dwRMSE;
%     dtrRMSE;

    %% Position RMSE
    posRMSE(seq+1, 1) = getRMSE(gt_pos - pr_pos);
    KFposRMSE(seq+1, 1) = getRMSE(gt_pos - kf_pos);

    %% Position RMSE per 100 iteration
    cutN = [1 2 3 4 5 6 7 8]*100;
    cutN_short = [1, 2]*100;
    if ~(seq==3 || seq==4)
        for ii = 1:1:8
            seqRMSE(ii) = getDriftN(pr_dtr, gt_dtr, cutN(ii));
            kfseqRMSE(ii) = getDriftN(kf_vel, gt_dtr, cutN(ii));
        end
        cnn_pc(seq+1) = mean(seqRMSE./cutN*100);
        kfcnn_pc(seq+1) = mean(kfseqRMSE./cutN*100);
    else
        kfseqRMSE = 0;
        seqRMSE = 0;
        for ii = 1:1:2
            seqRMSE(ii) = getDriftN(pr_dtr, gt_dtr, cutN_short(ii));
            kfseqRMSE(ii) = getDriftN(kf_vel, gt_dtr, cutN_short(ii));
        end
        seqRMSE
        cutN_short
        cnn_pc(seq+1) = mean(seqRMSE./cutN_short*100);
        kfcnn_pc(seq+1) = mean(kfseqRMSE./cutN_short*100);
    end
end
cnn_pc'
kfcnn_pc'

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


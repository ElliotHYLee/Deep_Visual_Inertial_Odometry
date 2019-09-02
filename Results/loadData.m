
%% Get Ground Truth Info.
getGT

%% Get Prediction Info.
% if noise ==0
%     getRefPR
% else
%     getPR
% end

getPR

%% Get Covariances
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

%% Concatenate Position
pr_pos = cumtrapz(pr_dtr_gnd);

%% read KF position


prPath = ['Data\',getPRPath(dsName, subType, seq)];
kfName = strcat(prPath, 'KF_pos', int2str(noise) ,'.txt')
kf_pos = importdata(kfName);


prPath = ['Data\',getPRPath(dsName, subType, seq)];
kfName = strcat(prPath, 'KF_vel', int2str(noise) ,'.txt')
kf_vel = importdata(kfName);
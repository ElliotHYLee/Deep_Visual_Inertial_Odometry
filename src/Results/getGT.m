gtPath = getGTPath(dsName,subType, seq);
gt_dtName = strcat(gtPath, 'dt.txt');

gt_duName = strcat(gtPath, '\du.txt');
gt_dwName = strcat(gtPath, '\dw.txt');
gt_dwGyroName = strcat(gtPath, '\dw_gyro.txt');
gt_dtrName = strcat(gtPath, '\dtrans.txt');
gt_dtr_gndName = strcat(gtPath, '\dtrans_gnd.txt');
gt_pos_gndName = strcat(gtPath, '\pos.txt');
% linRName = strcat(gtPath, '\linR.txt');
linRName = strcat(gtPath, '\linR', int2str(noise), '.txt');
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
prPath = ['Data\',getPRPath(dsName, subType, seq)];
pr_duName = strcat(prPath, '_du', int2str(noise), '.txt');
pr_dwName = strcat(prPath, '_dw', int2str(noise), '.txt');
pr_dtr_gndName = strcat(prPath, '_dtr_gnd', int2str(noise), '.txt');
pr_dtr_Name = strcat(prPath, '_dtr', int2str(noise), '.txt');
pr_duCovName = strcat(prPath, '_du_cov', int2str(noise), '.txt');
pr_dwCovName = strcat(prPath, '_dw_cov', int2str(noise), '.txt');
pr_dtrCovName = strcat(prPath, '_dtr_cov', int2str(noise), '.txt');


pr_duName;
pr_du = importdata(pr_duName);
pr_dw = importdata(pr_dwName);
pr_dtr = importdata(pr_dtr_Name);
pr_dtr_gnd = importdata(pr_dtr_gndName);
pr_du_cov = importdata(pr_duCovName);
pr_dw_cov = importdata(pr_dwCovName);
pr_dtr_cov = importdata(pr_dtrCovName);


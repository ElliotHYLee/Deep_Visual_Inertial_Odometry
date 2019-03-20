function[path] = getGTPath(dsName, subType, seq)
    dsName = lower(dsName);
    subType = lower(subType);
    if strcmp(dsName,'airsim')
       path = strcat('D:/DLData/Airsim/', subType, int2str(seq), '/');
    elseif strcmp(dsName,'euroc')
       path = strcat('D:/DLData/EuRoc/mh_', subType, int2str(seq), '/'); 
    elseif strcmp(dsName, 'kitti')
        if seq < 10
            path = strcat('D:\DLData\KITTI\odom\dataset\sequences\0', subType, int2str(seq), '/'); 
        else
            path = strcat('D:\DLData\KITTI\odom\dataset\sequences\', subType, int2str(seq), '/'); 
        end
    end
    

end
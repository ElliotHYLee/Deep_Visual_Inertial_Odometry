%this is mess. Hope I could fix it sometime.
function[path] = getGTPath(dsName, subType, seq)
    dsName = lower(dsName);
    subType = lower(subType);
    if strcmp(dsName,'airsim')
        if strcmp(subType, 'mr') || strcmp(subType, 'bar') || strcmp(subType, 'pin')|| strcmp(subType, 'edge')
            path = strcat('F:/DLData/Airsim/', 'mr', int2str(seq), '/');
        elseif strcmp(subType, 'mrseg')
            path = strcat('F:/DLData/Airsim/', 'mrseg', int2str(seq), '/');
        end
    elseif strcmp(dsName,'myroom')
        if strcmp(subType, 'none') || strcmp(subType, '')
            path = strcat('F:/DLData/MyRoom/data', int2str(seq), '/');
        end
    elseif strcmp(dsName,'mycar')
        if strcmp(subType, 'none') || strcmp(subType, '')
            path = strcat('F:/DLData/MyCar/data', int2str(seq), '/');
        end
    elseif strcmp(dsName,'agz')
        if strcmp(subType, 'none') || strcmp(subType, '')
            path = strcat('F:/DLData/AGZ/');
        end
    elseif strcmp(dsName,'euroc')
       path = strcat('F:/DLData/EuRoc/mh_',  int2str(seq), '/'); 
    elseif strcmp(dsName,'euroc_')
       path = strcat('F:/DLData/EuRoc_/mh_',  int2str(seq), '/'); 
    elseif strcmp(dsName, 'kitti')
        subType='';
        if seq < 10
            path = strcat('F:\DLData\KITTI\odom\dataset\sequences\0', subType, int2str(seq), '/'); 
        else
            path = strcat('F:\DLData\KITTI\odom\dataset\sequences\', subType, int2str(seq), '/'); 
        end
    end
    

end
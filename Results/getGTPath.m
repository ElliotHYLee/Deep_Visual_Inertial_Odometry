function[path] = getGTPath(dsName, subType, seq)
    dsName = lower(dsName);
    subType = lower(subType);
    if dsName=='airsim'
       path = strcat('D:/DLData/Airsim/', subType, int2str(seq), '/');
    end
    

end
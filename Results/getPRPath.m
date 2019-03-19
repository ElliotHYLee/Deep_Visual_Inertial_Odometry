function[path] = getPRPath(dsName, subType, seq)
    dsName = lower(dsName);
    subType = lower(subType);
    path = strcat(getBranchName(), '_', dsName, '_', subType, int2str(seq));
end
function[result] = getDriftN(gt, pr, cutN)
    idx = findIdx(gt, cutN);
    dist = sumDistUpto(gt, idx)
    needed = cutN - dist;
    pr100 = getPrDist(gt, pr, idx, needed);
    result = abs(pr100-cutN);
end


function[result] = getPrDist(gt, pr, idx, needed)
    prDist_cumPrev = sumDistUpto(pr, idx)
    prLastFrameDist = norm(pr(idx+1,:));
    gtLastFrameDist = norm(gt(idx+1,:));
    prLastFrameSeg = prLastFrameDist/gtLastFrameDist*needed;
    result = prDist_cumPrev + prLastFrameSeg;
end



function[idx] = findIdx(gt, cutN)
    N = length(gt);
    dist = 0;
    for i=1:1:N
       dist_frame = norm(gt(i,:));
       dist = dist + dist_frame;
       if (dist > cutN)
          idx = i - 1;
          break;
       end
    end
end


function[result] = sumDistUpto(gt, idx)
    dist = 0;
    for i=1:1:idx
       dist_frame = norm(gt(i,:));
       dist = dist + dist_frame;
    end
    result = dist;
end
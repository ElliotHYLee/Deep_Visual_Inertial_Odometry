function[Q, cov3] = getCov(chol)
    for i = 1:1:length(chol)
        row = chol(i,:);
        L = [row(1), 0     , 0;
             row(2), row(3), 0;
             row(4), row(5), row(6)];
        Q{i} = L*L';
        cov3(i,:) = diag(Q{i});
    end
end
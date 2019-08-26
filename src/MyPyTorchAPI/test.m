clc, clear, close all

gt_x = [1 2 3; 4 5 6]
pr_x = [0.8, 2.1, 3; 3.9, 5.2, 5.8]
Q = eye(3)

err = gt_x-pr_x

for i =1:1:2
    md(i,:) = err(i,:)*Q^-1*err(i,:)';
end
md


gt_x = [7,8,9;10,11,12]
pr_x = [6.6, 8.05, 9.11;9.985, 11.3, 11.9]
Q = eye(3)

err = gt_x-pr_x

for i =1:1:2
    md(i,:) = err(i,:)*Q^-1*err(i,:)';
end
md






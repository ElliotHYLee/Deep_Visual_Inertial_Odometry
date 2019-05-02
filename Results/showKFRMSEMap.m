clc, clear, close all
dsName = 'airsim';
subType = 'mr';
seq = 2;

loadData

time = cumtrapz(dt);
acc_gnd = dt.*acc_gnd;
vel_imu = cumtrapz(time, acc_gnd);

% KF
velKF = [0 0 0];
A = eye(3);
H = eye(3);
P{1} = eye(3)*10^-10;
% R = [1 0 0; 0 1 0; 0 0 1]*10^-5
% airsim mr


randN = 1000;
[RCell, xyz] = getRandRN(randN);

for iter=1:1:randN
    R = RCell{iter};
    for i=1:1:N
        velKF(i+1,:) = A*velKF(i,:)' + dt(i)*acc_gnd(i,:)';
        %R = acc_Q{i};
        pp = A*P{i}*A' + R;

        mCov = dtr_Q_gnd{i};
        K = pp*H'*inv(H*pp*H' + mCov);
        z = pr_dtr_gnd(i,:)';
        velKF(i+1,:) = (velKF(i+1,:)' + K*(z-H*velKF(i+1,:)'))';
        P{i+1} = pp - K*H*pp;
    end
    posKF = cumtrapz(velKF);
    posKF = posKF(2:end,:);
    posGT = gt_pos;
    RMSE(iter,:) = getRMSE3((posKF - posGT));
end

figure
subplot(3,1,1)
plot(xyz(:,1), RMSE(:,1), 'r.')
ylim([0, 10])
xlim([-10, 0])
subplot(3,1,2)
plot(xyz(:,2), RMSE(:,2), 'r.')
ylim([0, 10])
xlim([-10, 0])
subplot(3,1,3)
plot(xyz(:,3), RMSE(:,3), 'r.')
ylim([0, 10])
xlim([-10, 0])







function[R, xyz] = getRandRN(N)
    for i =1:1:N
        [RR, x, y, z] = getRandR();
        R{i} =  RR;
        xyz(i,:) = [x, y, z];
    end

end

function[R, x, y, z] = getRandR()
    a = getRandAB(-10, -1);
    b = getRandAB(-10, -1);
    c = getRandAB(-10, -1);
    sa = getSign();
    sb = getSign();
    sc = getSign();
    x = getRandAB(-10, -1);
    y = getRandAB(-10, -1);
    z = getRandAB(-10, -1);
    R = [10^x sa*10^a sb*10^b;
        sa*10^a 10^y sc*10^c;
        sb*10^b sc*10^c 10^z];
end

function[s] = getSign()
    if rand() > 0.5
        s = 1;
    else
        s = -1;
    end
end

function[r] = getRandAB(a, b)
r = (b-a).*rand(1,1) + a;
end

function[result] = getRMSE3(err)
    N = length(err);
    se = err.^2;
    sse = sum(se);
    mse = sse/N;
    rmse = sqrt(mse);    
    result = rmse;
end

function[result] = getRMSE(err)
    rmse3 = getRMSE3(err);
    result = mean(rmse3);
end

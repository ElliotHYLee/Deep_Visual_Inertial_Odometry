
function[x,y] = getELPS(Q, s)
[vec, val] = eig(Q);
val = diag(val);
[v, idx] = max(val);
V = vec(:,idx);
angle = atan(V(2)/V(1));
yaw = angle;
yaw*180/pi;

a = 2*sqrt(val(1)*s);
b = 2*sqrt(val(2)*s);

amp = max([a,b]);


t = linspace(0,2*pi,1000);
theta0 = yaw;
x = amp*sin(t+theta0);
y = amp*cos(t);
end

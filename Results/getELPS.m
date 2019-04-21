
function[x,y] = getELPS(Q, s)
[vec, val] = eig(Q);
val = diag(val);
[v, idx] = max(val);
V = vec(:,idx);

a = 2*sqrt(val(1)*s);
b = 2*sqrt(val(2)*s);
amp = max([a,b]);

t = linspace(0,2*pi,1000);
m = [vec [0;0];0 0 0 ];
eul = rotm2eul(m);
eul(1)*180/pi
theta0 = eul(1);%atan(V(2)/V(1));
x = amp*sin(t+theta0);
y = amp*cos(t);
end

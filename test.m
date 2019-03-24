clc, clear, close all


du = [1 2 3;4 5 6]
dw=du

thsq = dot(du,du,2)
th = sqrt(thsq)

A = sin(th)./th
B = (1-cos(th))./thsq
C = (1-A)./thsq

skew1 = make_skew(dw(1,:))
skew2 = skew1*skew1;
eye(3) + B(1,:)*skew1 + C(1,:)*skew2

skew1 = make_skew(dw(2,:))
skew2 = skew1*skew1;
eye(3) + B(2,:)*skew1 + C(2,:)*skew2




function[sk] = make_skew(w)
    sk = [0     -w(3)  w(2) ;...
          w(3)  0      -w(1) ;...
         -w(2)  w(1)   0 ]; 
end






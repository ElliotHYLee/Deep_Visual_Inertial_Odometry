classdef LieSO3 < Lie
   methods
       function[dR] = getExp(obj, w)
           th = sqrt(w'*w);
           wx = obj.make_skew(w);
           if th==0
               dR = eye(3);
           else
               dR = eye(3) + sin(th)/th*wx  + (1-cos(th))/th^2*wx*wx;
           end
       end
       
       function[skew] = getLog(obj, rot)
           th = acos((trace(rot)-1)/2);
           if th == 0
               skew = zeros(3,3);
           else
               skew = th/(2*sin(th))*(rot-rot');
           end
       end
   end
end
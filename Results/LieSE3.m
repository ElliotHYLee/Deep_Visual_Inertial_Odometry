classdef LieSE3 < Lie
   properties
      SO3 = LieSO3(); 
   end
   methods
       function[R,t] = getRt(obj,T)
            R = T(1:3,1:3);
            t = T(1:3,4);
       end
       
       function[T] = getExp(obj, w, u)
           V = obj.get_V(w);
           t = V*u;
           dR = obj.SO3.getExp(w);
           T = [dR,t;zeros(1,3),1];
       end
       
       function[V] = get_V(obj, w)
           th = sqrt(w'*w);
           skew = obj.make_skew(w);
           if th==0
               V = eye(3) ;%+ 1/2*skew + 1/6*skew*skew;
           else
               V = eye(3) + (1-cos(th))/th^2*skew +(th-sin(th))/th^3*skew*skew;
           end
        end
       
       function[w, u] = getLog(obj, T)
           [R,t] = obj.getRt(T);
           skew = obj.SO3.getLog(R)
           w = obj.get_w(skew)';
           V = obj.get_V(w)
           u = V^-1*t;
       end
   end
end
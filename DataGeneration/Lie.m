classdef Lie
   methods
       function[w] = get_w(obj, skew)
         w = [-skew(2,3), skew(1,3), -skew(1,2)];
        end

       function[sk] = make_skew(obj, w)
            sk = [0     -w(3)  w(2) ;...
                  w(3)  0      -w(1) ;...
                 -w(2)  w(1)   0 ]; 
       end
   end
end
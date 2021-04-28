function mtx = dRTdq()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
mtx = [1 0 0 0
        0 -1 0 0
        0 0 -1 0
        0 0 0 -1
        
        0 1 0 0
        1 0 0 0
        0 0 0 1
        0 0 -1 0
        
        0 0 1 0
        0 0 0 -1
        1 0 0 0
        0 1 0 0
        
        0 0 0 1
        0 0 1 0
        0 -1 0 0
        1 0 0 0];
end
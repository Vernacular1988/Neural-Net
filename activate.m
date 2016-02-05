function [ a ] = activate( W, x,b)
%ACTIVATE Summary of this function goes here
%   Detailed explanation goes here
s=W*x+b;

a=sigmf(s,[1 0]);

end


function [a_next, z_next] = fc(w, a, x, f)
%FC Manual forward computation for one fully connected layer.
%   Samples are columns. x contains any external input and a contains the
%   previous layer activation. For the conventional MLP, pass x=[] after
%   the first layer.

input = [x; a];
z_next = w * input;
a_next = f(z_next);
end

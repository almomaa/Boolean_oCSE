function [E,U,P] = dEntropy(X)

% Entropy estimator of a discrete random variable
%
% Inputs:
%    X: is an n-by-d symbolic matrix (that can be integers or characters)
%       each row in X represent a state in d-dimensional space
 
% Outputs:
%    E: Entropy of X
%    U: Unique states (rows) in the data X
%    P: Probablity of occurence of each unique state in U

[U,~,ic] = unique(X,'rows');
P = accumarray(ic,1)./length(ic);
E = -dot(P,log2(P));
end
function [IX, Table] = BoCSE(y,X,varargin)

% Boolean optimal Causation Entropy
%
% This function extract the optimal subset from X that influnce 
% the dynamic of y
%
%% inputs:
%         y: n-by-1 discrete variable.
%         X: n-by-d discrete matrix.
%
%       A set of options can be entered as a pair inputs.
%       these options are
%
%  * 'keepin': a set of index of columns of X that we wish to keep (if
%       any). example call 
%               - BoCSE(y,X,'keepin',[]) (default)
%               - BoCSE(y,X,'keepin',[1 3 4])
%
%  * 'alpha': Confedence parameter (0 < alpha <=1) for the shuffle test.
%             The default value is alpha = 0.98. example call:
%               - BoCSE(y,X,'alpha',0.9)
%
%  * 'numPer': number of pernutations for the shuffle test. can be any
%              integer > 0. the default is numPer = 200. example call:
%               - BoCSE(y,X,'numPer',500)
%
%   The 'input pairs' can be combined. Sample call:
%       - BoCSE(y,X)
%       - BoCSE(y,X,'alpha',0.99,'numPer',1000)
%       - BoCSE(y,X,'keepin',[2 3],'numPer',1000)
%       - BoCSE(y,X,'numPer',500,'alpha',0.925,'keepin',[5])
%
%% Outputs:
%
%   index: vector of indecies of columns of X that influnce y.
%
%   Table: a table that has the unique states found for the optimal
%   subset of X. and the probability of occurence of each one of the unique
%   states.


%% Read Inputs and Initialization
p = inputParser;
p.addRequired('y');
p.addRequired('X');

p.addParameter('keepin',[]);
p.addParameter('alpha',0.98,@(x) x>0 && x<=1);
p.addParameter('numPer',500,@isscalar);

p.parse(y,X,varargin{:});
options = p.Results;



dim = size(X,2);
IX = false(1,dim);
IX(options.keepin) = true;


%% Forward Selection and Backward Elimination of BoCSE
IX = forward(y,X,IX,options);
IX = backward(y,X,IX,options);


%% Extract Outputs
index = find(IX);

if isempty(index)
    Table = 'No relevent variable found';
    return;
end
varnames = cell(1,length(index)+2);
for i=1:length(index)
    varnames{i} = ['X' num2str(index(i))];
end
% varnames{end-2} = 'Probability X={x_i:i\in index}';
varnames{end-1} = 'Probability Y=0';
varnames{end} = 'Probability Y=1';

states = unique(X(:,index),'rows');
P = zeros(size(states,1),2);
for i=1:size(states,1)
    ix = ismember(X(:,index),states(i,:),'rows');
%     P(i,1) = sum(ix)/length(ix);
    P(i,1) = (sum(~y(ix))/sum(ix));
    P(i,2) = (sum( y(ix))/sum(ix));
end

% [~,class] = max(P(:,2:3),[],2);
% class = class-1;

% IX = ismember([X(:,index), y],[states, class],'rows');

% PE = 1 - sum(ix)/length(ix)
% [~,U,P] = dEntropy([X(:,index), y]);


Table = array2table([states P],'VariableNames',varnames);
end


function IX = forward(y,X,IX,options)
% Forward Greedy Feature Selection
dim = size(X,2);
Done = false;

% [~,Is] = shuffleTest(y,y,[],0,options);
% tol = max(Is,0.001);

while ~Done
    I = -inf(1,dim);
    for i=1:dim
        if ~IX(i)
            I(i) = dCMI(y,X(:,i),X(:,IX));
        end
    end
    [~,ix] = max(I);

    Done = ~shuffleTest(y,X(:,ix),X(:,IX),I(ix),options);
    if ~Done
        IX(ix) = true;
%     else
%         Done = true;
    end
end

end

function IX = backward(y,X,IX,options)
% Backward Greedy Feature Elimination
dim = size(X,2);
Done = false;

% [~,Is] = shuffleTest(y,y,[],0,options);
% tol = min(Is,0.001)

while ~Done
    I = inf(1,dim);
    for i=1:dim
        if IX(i) && ~ismember(i,options.keepin)
            mask = IX;
            mask(i) = false;
            I(i) = dCMI(y,X(:,IX),X(:,mask));
        end
    end
    [~,ix] = min(I);

    mask = IX;
    mask(ix) = false;
%     I(ix)
%     find(IX)
%     pause

    if I(ix)>1e-3
        Done = shuffleTest(y,X(:,ix),X(:,mask),I(ix),options);
    end

    if ~Done
        IX(ix) = false;
        Done = false;
    end
end
end

function [success, I] = shuffleTest(y,x,Z,val,options)
% Shuffle test of the conditional mutual information: I(X;Y|Z)
%  H0 (null): I(X;Y|Z)=0
%  H1 (alternative): I(X;Y|Z)>0

% Input
%    x: n-by-nx, n samples of x
%    y: n-by-ny, n samples of y
%    z: n-by-nz, n samples of Z
%    val: estimated CMI value
%  options: a struct variable that has the variables:
%           options.alpha: alpha-level
%           options.numPer: number of shuffles to perform

% Output
%   success : Boolean variable 
%             success = true  ==> accept H1
%             success = false ==> accept H0

I = zeros(1,options.numPer);
alpha = options.alpha;
for i=1:length(I)
    I(i) = dCMI(y,x(randperm(length(x)),:),Z);
end
I = sort(I);
I = I(ceil(alpha*length(I)));
if val>I
    success = true;
else
    success = false;
end

end

function I = dMI(x,y)
% Mutual Information estimator of a discrete random variable
%
% Inputs:
%    x: is an n-by-d1 symbolic matrix (that can be integers or characters)
%       n points in the d-dimensional sample space
%    y: is an n-by-d2 symbolic matrix (that can be integers or characters)
%       n points in the d-dimensional sample space
%
 
% Outputs:
%   I = mutual information between x and y.

    I = dEntropy(x) + dEntropy(y) - dEntropy([x,y]);
end

function I = dCMI(x,y,z)
% Conditional Mutual Information estimator of a discrete random variable
%
% Inputs:
%    x: is an n-by-d1 symbolic matrix (that can be integers or characters)
%       n points in the d-dimensional sample space
%    y: is an n-by-d2 symbolic matrix (that can be integers or characters)
%       n points in the d-dimensional sample space
%    z: is an n-by-d3 symbolic matrix (that can be integers or characters)
%       n points in the d-dimensional sample space. 
%
%    The function can be called with empty conditioning set (dCMI(x,y,[])) to find the
%    mutual information. Providing the third input variable (dCMI(x,y,z))
%    will find the conditional mutual information (the information between
%    x and y, given z).
 
% Outputs:
%   I = mutual information (if called with two variables).
%   I = conditional MI (if called with three variables).
if isempty(z)
    I = dMI(x,y);
    return;
end
    I = dEntropy([x,z]) + dEntropy([y,z]) - dEntropy(z) - ...
        dEntropy([x,y,z]);
end


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


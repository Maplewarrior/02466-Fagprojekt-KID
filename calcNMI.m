function NMI=calcNMI(Z1,Z2)

% Function to calculate normalized mutual information between two assignment matrices
% Z1 and Z2
%
% Usage:
%  MI=calcMI(Z1,Z2)
%
% Input:
%   Z1  D1 x N Assignment matrix
%   Z2  D2 x N Assignment matrix
%   
% Output:
%   NMI Normalized Mutual information between Z1 and Z2.
%
% Written by Morten M�rup

NMI=(2*calcMI(Z1,Z2))/(calcMI(Z1,Z1)+calcMI(Z2,Z2));

# z1 = [2,4;5,4]
# z2 = [1,7;4,3]
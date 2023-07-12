function H = channel(N,d,a,type,variance)

% Calculate the channel matrix (path-loss + fading)
% H = channel(M,N,d,a)
%
% M: Number of PUs
% N: Number of SUs
% d: Euclidean distance between each PU-SU pair
% a: Path-loss exponent
H=zeros(N,1);

if (strcmp(type,'ray'))
    H=sqrt(-2 * variance * log(rand([N,1])))/sqrt(2);

elseif (strcmp(type,'nakagami'))
    m=1.5;
    omega=1;
    H = sqrt(gamrnd(m,omega/m,[N,1])).*sqrt(variance);

elseif (strcmp(type,'rician'))
    sigma = sqrt(variance);
    H = sigma*sqrt(randraw('chisqnc', [(1/sigma)^2, 2], [N,1]));
    % H=H.*sqrt(variance);
else
    H = ones(N,1);
end
H = H.*sqrt(d.^(-a));% Fading + path-loss (amplitude loss)
end

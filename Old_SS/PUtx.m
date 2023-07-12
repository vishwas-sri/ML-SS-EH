function [X,S] = PUtx(samples,txPower, Pr)

% Get the PU transmission
%
% PUtx(M,samples,txPower, Pr_1)
% M - Number of PUs
% samples - Number of transmission samples
% txPower - Average transmission power for each PU
% Pr - Active probability for each PU

S = zeros(1); % PU states
X = zeros(samples,1); % Signal at PU transmitters

    p = rand(1);
    if (p <= Pr)
        S(:) = 1;
    end


if (S>0)
    X = normrnd(zeros(samples,1),ones(samples,1).*sqrt(txPower));
end

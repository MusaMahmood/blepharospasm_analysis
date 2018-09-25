function [ E ] = get_intensity( X )
%GET_INTENSITY Summary of this function goes here
%   Detailed explanation goes here

% [psd, f] = welch_psd(X, 250, hannWin(length(X)));
% Y = rescale_minmax(psd);
% waveletcoef = cwt(X',1:20,'haar');
waveletcoef = cwt_haar(X');
S = abs(waveletcoef.*waveletcoef); % Obtain Scalogram
% figure(2); imagesc(S);
E = sum(S(:)); % Obtain Energy Level

end


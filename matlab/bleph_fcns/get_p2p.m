function [ p2p ] = get_p2p( X )
%GET_P2P
% Input raw 2000, 1 array
% runs filter [bh, ah] = butter(3, 1*2/250, 'high'); % Fs = 250 Hz for this one.
% gets p2p
bh = [0.975179811634754,-2.925539434904263,2.925539434904263,-0.975179811634754];
ah = [1,-2.949735839706348,2.900726988355438,-0.950975665016249];
y = filtfilt(bh, ah, X);
p2p = peak2peak(y); 


end


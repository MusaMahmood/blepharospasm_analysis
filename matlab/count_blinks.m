function [ count_peaks ] = count_blinks( X )
% count_blinks:

[pk, loc] = findpeaks(X, 'MinPeakProminence', 0.3339);
count_peaks = length(pk);




end


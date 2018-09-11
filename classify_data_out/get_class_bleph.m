function [ index, scores ] = get_class_bleph( input_array )
%get_class_bleph - input a 
% input [2000, 3] class array

scores = single(sum(input_array));
[~, index] = max(scores);

index = index-1;

end


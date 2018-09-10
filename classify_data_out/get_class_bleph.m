function [ index, scores ] = get_class_bleph( input_array )
%get_class_bleph - input a 
% input [2000, 4] class array
scores = single(sum(input_array));
[m, index] = max(scores);

index = index-1;

end


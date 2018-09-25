function [ rearranged ] = rearrange_5c( input_array )
rearranged = single(zeros(2000, 5));
for i = 1:2000
    for j = 1:5
        rearranged(i, j) = input_array((i-1)*5 + j);
    end
end

end


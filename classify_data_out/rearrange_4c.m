function [ rearranged ] = rearrange_4c( input_array )
rearranged = single(zeros(2000, 4));
for i = 1:2000
    for j = 1:4
        rearranged(i, j) = input_array((i-1)*4 + j);
    end
end

end


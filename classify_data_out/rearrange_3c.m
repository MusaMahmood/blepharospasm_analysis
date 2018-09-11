function [ rearranged ] = rearrange_3c( input_array )
rearranged = single(zeros(2000, 3));
for i = 1:2000
    for j = 1:3
        rearranged(i, j) = input_array((i-1)*3 + j);
    end
end

end


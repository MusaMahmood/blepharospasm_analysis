% analyze_classification
data = csvread('classification_outputs\EOG_classify0.csv');
data(:, [7:8]) = rand(60, 2);
data_concat = [data(:, 1); data(:, 2); data(:, 3); data(:, 4); data(:, 5); data(:, 6); data(:, 7); data(:, 8)];
[ count, symptoms ] = bleph_analyze_data(single(data_concat));

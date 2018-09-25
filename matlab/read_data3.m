% read_data3
% New data from SJ simulation test. 
clr;
Fs = 250;
[b, a] = butter(3, 1*2/Fs, 'high');
normals = {'EOGData_2018.09.19_14.00.41_250Hz.csv', 'EOGData_2018.09.19_14.02.15_250Hz.csv'};
% Normal Data: (?) 
%     1. '
%     2. 
%     3. 
% Pathological Blinking?: EOGData_2018.09.19_14.17.39_250Hz.csv'
% Blepharospasm: EOGData_2018.09.19_14.14.50_250Hz.csv'
data = csvread(['dataset\sj_demo_data\' normals{1}]);
t = data(:, 1);
data = data(:, 2);
fdata = filtfilt(b, a, data);
plot(t, fdata);
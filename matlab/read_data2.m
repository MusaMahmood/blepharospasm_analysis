clr;
Fs = 250; PLOT = 0; SAVE = 1;
[b, a] = butter(3, 1*2/Fs, 'high');
%{  
    Labels:
    0 - Normal
    1 - Involuntary Blinking
    2 - Apraxia (severe)
    3 - Eyes forced closed.
%}
%%% RECORDED WITH nRF/ADS1292: %%%
    % NORMAL BLINKING %
set1 = csvread('dataset\musa_normal\EOGData_2018.09.10_12.41.19_250Hz.csv'); set1 = set1(:, 2);
labels{1} = zeros(length(set1), 1);
set2 = csvread('dataset\musa_normal\EOGData_2018.09.10_12.46.23_250Hz.csv'); set2 = set2(:, 2);
labels{2} = zeros(length(set2), 1);
set3 = csvread('dataset\musa_normal\EOGData_2018.09.10_12.50.33_250Hz.csv'); set3 = set3(:, 2);
labels{3} = zeros(length(set3), 1);
    % WITHOUT NORMAL 2D EYE MOVEMENT - SINGLE PT FOCUS %
set4 = csvread('dataset\musa_normal\EOGData_2018.09.10_13.02.37_250Hz.csv'); set4 = set4(:, 2);
labels{4} = zeros(length(set4), 1);
%%% RECORDED WITH BIORADIO %%%
    % APRAXIA %
set5 = csvread('dataset\Patient1_apraxia_severe.csv'); set5 = set5(1:89730, 1);
labels{5} = 3*ones(89730, 1);
    % PATHOLOGICAL BLINKING %
set6 = csvread('dataset\Patient2_increased_blink.csv'); set6 = -set6(:, 1);
labels{6} = 1*ones(length(set6), 1);
set7 = csvread('dataset\Patient3_increased_blink.csv'); set7 = set7(:, 1);
labels{7} = 1*ones(length(set7), 1);
%%% RECORDED WITH nRF/ADS1292 %%%
    % EYE SPASM (SIMULATED)
        % INTENSITY LEVEL 1 %
set8 = csvread('dataset\musa_simulated_spasm\EOGData_2018.09.10_13.49.20_250Hz.csv'); set8 = set8(:, 2);
labels{8} = 2*ones(length(set8), 1);
labels{8}(10380:11560) = 0;
        % INTENSITY LEVEL 2 %
set9 = csvread('dataset\musa_simulated_spasm\EOGData_2018.09.10_13.51.41_250Hz.csv'); set9 = set9(:, 2);
labels{9} = 2*ones(length(set9), 1);
        % INTENSITY LEVEL 3 %
set10 = csvread('dataset\spasm2.csv'); set10 = set10(:, 1);
labels{10} = zeros(length(set10), 1);
labels{10}(1:2814) = 0;
labels{10}(2814:end) = 2;
set11 = csvread('dataset\spasm5.csv'); set11 = set11(:, 1);
labels{11} = zeros(length(set11), 1);
labels{11}(1:2955) = 0;
labels{11}(2956:end) = 2;

data_combined = {set1, set2, set3, set4, set5, set6, set7, set8, set9, set10, set11};
L = cell2mat(labels');
%% Window Out Everything:
for d = 1:length(data_combined)
    set = data_combined{d};
    whop = Fs*1; wlen = Fs*8; wStart = 1:whop:(length(set)-wlen); wEnd = wStart + wlen - 1;
    Y = zeros(length(wStart), wlen, 1);
    Yp2p = zeros(length(wStart), 1);
    windows_raw = zeros(length(wStart), wlen, 1);
    for w = 1:length(wStart)
        fprintf('[%d, %d/%d ] \n', d, w, length(wStart));
        filt_sig = filtfilt(b, a, set(wStart(w):wEnd(w), 1));
        Yp2p(w) = get_p2p(set(wStart(w):wEnd(w), 1));
        windows_raw(w, :, 1) = rescale_minmax(filt_sig);
        Y(w, :, :) = labels{d}( wStart(w):wEnd(w) );
        if PLOT
            figure(1); subplot(2,1,1);plot(squeeze(windows_raw(w, :, 1)));
            subplot(2,1,2); plot(squeeze(Y(w, :, :)));
            con = input('Continue ?\n');
        end
    end
    Yp2pmin(d) = min(Yp2p);
    Yp2pmax(d) = max(Yp2p);
    if SAVE
        relevant_data = windows_raw;
        out_d = 'ml_bleph/data_labeled2/';
        mkdir(out_d); 
        fn = ['set', num2str(d), '.mat'];
        save([out_d, fn], 'relevant_data', 'Y'); clear relevant_data windows_raw Y
    end
end
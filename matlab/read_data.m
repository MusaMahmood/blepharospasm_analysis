%{  
    Labels:
    0 - Normal
    1 - Involuntary Blinking
    2 - Apraxia (severe)
    3 - Eyes forced closed.
%}
clr;
Fs = 250; PLOT = 0; SAVE = 0;
[bh, ah] = butter(3, 1*2/Fs, 'high');

set1 = csvread('dataset\Patient1_apraxia_severe.csv'); set1 = set1(1:89730, 1);
labels{1} = 2*ones(89730, 1);
set2 = csvread('dataset\Patient2_increased_blink.csv'); set2 = -set2(:, 1);
labels{2} = 1*ones(length(set2), 1);
set3 = csvread('dataset\Patient3_increased_blink.csv'); set3 = set3(:, 1);
labels{3} = 1*ones(length(set3), 1);

normal = csvread('dataset\blink5.csv'); normal = normal(:, 1);
labels{4} = zeros(length(normal), 1);

bspasm1 = csvread('dataset\spasm2.csv'); bspasm1 = bspasm1(:, 1);
labels{5} = zeros(length(bspasm1), 1);
labels{5}(1:2814) = 0;
labels{5}(2814:end) = 3;

bspasm2 = csvread('dataset\spasm5.csv'); bspasm2 = bspasm2(:, 1);
labels{6} = zeros(length(bspasm2), 1);
labels{6}(1:2955) = 0;
labels{6}(2956:end) = 3;

temp = csvread('dataset\EOG_classify_2018.09.06_13.38.14_250Hz.csv');
labels{7} = zeros(length(temp), 1);

data_combined = {set1, set2, set3, normal, bspasm1, bspasm2, temp};

for d = 1:length(data_combined)
    set = data_combined{d};
    whop = Fs*1; wlen = Fs*8; wStart = 1:whop:(length(set)-wlen); wEnd = wStart + wlen - 1;
    Y = zeros(length(wStart), wlen, 1);
    Yp2p = zeros(length(wStart), 1);
    windows_raw = zeros(length(wStart), wlen, 1);
    for w = 1:length(wStart)
        fprintf('[%d, %d/%d ] \n', d, w, length(wStart));
        filt_sig = filtfilt(bh, ah, set(wStart(w):wEnd(w), 1));
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
        out_d = 'ml_bleph/data_labeled/';
        mkdir(out_d); 
        fn = ['set', num2str(d), '.mat'];
        save([out_d, fn], 'relevant_data', 'Y'); clear relevant_data windows_raw Y
    end
end

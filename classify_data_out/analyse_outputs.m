clr;
% load('k_bleph_annotate\k_bleph_annotate_temp.mat');
% load('k2_bleph_annotate\k2_bleph_annotate_v.mat');
load('k3_bleph_annotate\k3_bleph_annotate.mat');
% TEMP = csvread('EOG_classify_2018.09.06_13.38.14_250Hz.csv');
% TEMP2 = TEMP(1:2000, 2:5);
% TEMP3 = TEMP2(:);
% for i = 1:2000
%     for j = 1:4
%         tfout(i, j) = TEMP3((i-1)*4 + j);
%     end
% end
% T3_r = rearrange_4c(single(TEMP3));
% [index, scores] = get_class_bleph(T3_r);
% %{
CH1_ONLY = size(x_val, 3) - 1; 
PLOT = 0;
score = 0; miss = 0;
acc = zeros(size(x_val,1), 1);
samples = size(y_val, 1);
number_classes = size(y_val, 3);

%%% TEMP:
tmp = zeros(2000, 3);
tmp_rrng = rearrange_3c(single(tmp));
tmp4 = zeros(2000, 4);
tmp_rrng4 = rearrange_4c(single(tmp4));
tmp5 = single(zeros(2000, 5));
tmp_rrng5 = rearrange_5c(tmp5);
tmp_getc = get_class_distribution(tmp5);

if exist('x_val', 'var')
    ytrue_all = vec2ind(reshape(y_val, [samples*2000,number_classes])' );
    ytest_all = vec2ind(reshape(y_out, [samples*2000,number_classes])' );
    acc = sum(ytest_all == ytrue_all)/length(ytrue_all);
    for s = 1:size(x_val,1)
        y_sample = squeeze(y_out(s, :, :));
        ysi = vec2ind(y_sample');
        y_true = squeeze(y_val(s, :, :));
        yti = vec2ind(y_true');
        syy = ysi == yti; pct = sum(syy)/size(y_val, 2);
        fprintf('Sample #: %d  Acc: %1.3f \n', s, pct); 
        if pct >= 0.8
            score = score + 1;
            acc(s) = 1;
        else
            miss = miss + 1;
            acc(s) = 0;
%             PLOT = 1;
        end
        clear b yy yt
        A = squeeze(y_out(s, :, :));
        [I, scores] = get_class_bleph(A);
        disp(scores);
        disp(I);
        if PLOT
%             PLOT = 0;
            sample = squeeze(x_val(s, :, :));
            [~, yl2] = max(y_sample, [], 2);
            figure(2); subplot(4, 1, 1); plot(sample); title('Data');
            subplot(4, 1, 2); plot(squeeze(y_prob(s, :, :))); title('y prob');
            subplot(4, 1, 3); plot(squeeze(y_out(s, :, :))); title('y out');
            subplot(4, 1, 4); plot(squeeze(y_val(s, :, :))); title('y true vals');
            xca = input('A1 Continue ? \n'); 
        end
        %assess accuracty
        
    end
    fprintf('Correct: %d, Miss %d \n', score, miss); 
end
s = size(y_val);
y_true = reshape(y_val, [s(1)*s(2), number_classes]);
y_pred = reshape(y_out, [s(1)*s(2), number_classes]);
sum(y_true)
sum(y_pred)
y_true = vec2ind(y_true');
y_pred = vec2ind(y_pred');
C = confusionmat(y_true, y_pred);
figure(1); conf_heatmap(C);
%}
%{
if exist('x_flex', 'var')
    for s = 322:2:size(x_flex,1)  %5206
        fprintf('Sample #: %d \n', s); 
        sample = squeeze(x_flex(s, :, :));
        figure(2);
        subplot(3, 1, 1); plot(sample); title('Data');
        subplot(3, 1, 2); plot(squeeze(y_prob_flex(s, :, :))); title('y prob');
        subplot(3, 1, 3); plot(squeeze(y_out_flex(s, :, :))); title('y out');
        xca = input('A1 Continue ? \n'); 
    end
end
s = size(y_out_flex);
y_pred = reshape(y_out_flex, [s(1)*s(2), 5]);
sum(y_pred)
%}
%{
if ~CH1_ONLY
    for s = 1:size(x_val,1)
        sample = squeeze(x_val(s, :));
        y_label_real = squeeze(y_val(s, :, :));
        y_label_predict = squeeze(y_out(s, :, :));
        figure(1); subplot(3, 1, 1); plot(sample);hold on;
        subplot(3, 1, 2); plot(y_label_real);title('Real Annot');
        subplot(3, 1, 3); hold on; plot(y_label_predict); title('Predicted Annot');
        xca = input('Continue ? \n'); c
lf(1);
    end
else
    for s = 1:size(x_val,1)
        sample = squeeze(x_val(s, :, :));
        y_label_real = squeeze(y_val(s, :, :));
        y_label_predict = squeeze(y_out(s, :, :));
        figure(1); subplot(2, 1, 1); plot(sample(:, 1)); subplot(2, 1, 2); plot(sample(:, 2));
        subplot(2, 1, 1); hold on; plot(y_label_real);title('Real Annot');
        subplot(2, 1, 2); hold on; plot(y_label_predict); title('Predicted Annot');
        xca = input('Continue ? \n'); clf(1);
    end
end
%}

% 1ch version
function [y] = within_n_percent(x, x2, n)
    %check if x1 within n of x2
    y = false;
    x_up = n*x + x;
    x_down = x - n*x;
    y = x2 < x_up & x2 > x_down;
end

function [y] = within_n_points(x, x2, n)
    y = false;
    x_up = x+n;
    x_down = x-n;
    y = x2 < x_up & x2 > x_down;
end
function [ count, symptoms ] = bleph_analyze_data( classifyIn_concat )
%bleph_analyze_data Creates a summary from classification data
% Input = (60, 8) in the form of [480, 1] concatenated together:
classifyIn = zeros(60, 8);
classifyIn = reshape(classifyIn_concat, [60, 8]);
% Columns are as follows:
%    1   |    2    |    3    |    4    |      5      |    6     |       7       |    8                              
% p2p mV | class 1 | class 2 | class 3 | outputClass | Severity | Number Blinks | Wavelet Energy (Haar mother wavelet)
% we only really need 1, 5:8, rest is exported for good measure
% relevant_data = classifyIn(:, [~,5,6,7,8]);

symptoms = zeros(4, 1); %  [ {Condition}, {Severity(1)}, {Severity(2)},  {Apraxia(probability)} ]
count = zeros(3, 1); % Counts for each of the classes

classes = classifyIn(:, 5);
severities = classifyIn(:, 6); 
% number_peaks = classifyIn(:, 7); 
% energy = classifyIn(:, 8);

idx = false(60, 8);
idx(:, 1) = classes == 0; % Normal
idx(:, 2) = classes == 1; % Pathological Blink
idx(:, 3) = classes == 2; % Blepharospasm

% Separate Severities into Relevant classes
severities_path = severities(idx(:, 2));
severities_blepharospasm = severities(idx(:, 3));

for i = 1:3
    count(i) = sum(idx(:, i));
end

% Primary Output Class:
[~, primary_class] = max(count); 

symptoms(1) = primary_class - 1; % Convert to zero-index

if (primary_class == 2)
    % Pathological Blinking
    symptoms(2) = mean(severities_path);
elseif primary_class == 3
    symptoms(3) = mean(severities_blepharospasm);
    % Blepharospasm
end

end


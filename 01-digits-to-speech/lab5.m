% LAB5 Manual MATLAB implementation of the handwritten-digit-to-speech MLP.
% No Deep Learning Toolbox functions are used.
clear; close all; clc;
rng(2023521460);

load mnist_small_matlab.mat
input_size = 28 * 28;
train_size = size(trainData, 3);
test_size = size(testData, 3);
X_train = single(reshape(trainData, input_size, train_size)) / 255;
X_test = single(reshape(testData, input_size, test_size)) / 255;

sample_rate = 4000;
audio = zeros(2983, 10, 'single');
for digit = 0:9
    [signal, fs] = audioread(fullfile('audio', sprintf('%d.wav', digit)));
    assert(fs == sample_rate, 'Expected a 4 kHz WAV file.');
    assert(numel(signal) == 2983, 'Every WAV file must contain 2,983 samples.');
    audio(:, digit + 1) = single((signal(:) + 1) / 2);
end

layer_size = [784 512 512 1024 2048 2983];
alpha = 0.1;
mini_batch = 100;
max_iter = 300;
L = numel(layer_size);
sigm = @(s) 1 ./ (1 + exp(-max(min(s, 30), -30)));
dsigm = @(s) sigm(s) .* (1 - sigm(s));

% Xavier/Glorot uniform initialization specified in the teacher's slides.
w = cell(L - 1, 1);
for l = 1:L-1
    limit = sqrt(6 / (layer_size(l) + layer_size(l + 1)));
    w{l} = single((rand(layer_size(l + 1), layer_size(l)) * 2 - 1) * limit);
end

J = zeros(max_iter, 1);
for iter = 1:max_iter
    ind = randperm(train_size, mini_batch);
    a = cell(L, 1); z = cell(L, 1); delta = cell(L, 1);
    a{1} = X_train(:, ind);
    y = get_audio(trainLabels(:, ind), audio);

    for l = 1:L-2
        [a{l + 1}, z{l + 1}] = fc(w{l}, a{l}, [], sigm);
    end
    z{L} = w{L - 1} * a{L - 1};
    a{L} = z{L}; % linear output
    error = a{L} - y;
    J(iter) = mean(error(:).^2);

    % MSE derivative, averaged across all 2,983 output samples.
    delta{L} = (2 / layer_size(end)) * error;
    for l = L-1:-1:2
        delta{l} = bc(w{l}, z{l}, delta{l + 1}, dsigm);
    end
    for l = 1:L-1
        gw = (delta{l + 1} * a{l}') / mini_batch;
        w{l} = w{l} - alpha * gw;
    end
    fprintf('%d/%d iterations: MSE = %.8f\n', iter, max_iter, J(iter));
end
save model.mat w layer_size J

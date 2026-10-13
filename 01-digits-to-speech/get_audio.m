function audio_y = get_audio(y, audio)
%GET_AUDIO Convert digit labels to 2,983-sample target waveforms.
%   audio must be 2983-by-10. y may be a 10-by-N one-hot matrix or a
%   vector of zero-based digit labels.

if isvector(y)
    labels = double(y(:)') + 1;
else
    [~, labels] = max(y, [], 1);
end
audio_y = audio(:, labels);
end

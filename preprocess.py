import numpy as np
from scipy import signal
import librosa

class AudioPreprocessor:
    def __init__(self, mu=0.9375, N=256, M=80, window_type='hamming', sample_rate=16000):
        self.mu = mu
        self.N = N
        self.M = M
        self.window_type = window_type
        self.sample_rate = sample_rate
        self.b = [1, -mu]
        self.a = 1
        self.window = signal.get_window(window_type, N)

    def mel_filterbank(self, num_filters):
        mel_filters = librosa.filters.mel(sr=self.sample_rate, n_fft=self.N, n_mels=num_filters)
        return mel_filters.T  # Transpose to match the shape used in the previous code snippet

    def preprocess_audio(self, wav):
        # Apply pre-emphasis filter
        wav = signal.lfilter(self.b, self.a, wav)

        # Frame division with overlap
        frames = []
        start = 0
        while start + self.N < len(wav[0]):
            frames.append(wav[0][start:start+self.N])
            start += self.N - self.M

        # Apply window function
        windowed_frames = [frame * self.window for frame in frames]

        # FFT transformation
        fft_frames = [np.fft.fft(frame) for frame in windowed_frames]

        # Triangle filterbank
        num_filters = 24
        mel_filters = self.mel_filterbank(num_filters)

        # Apply triangle filters
        filtered_frames = []
        for frame in fft_frames:
            filtered_frame = []
            for mel_filter in mel_filters:
                frame_reshaped = np.expand_dims(frame, axis=1)
                filtered_frame.append(np.sum(np.abs(frame_reshaped) ** 2 * mel_filter))
            filtered_frames.extend(filtered_frame)

        return np.array(filtered_frames)


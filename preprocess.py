import numpy as np
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
        self.window = librosa.filters.get_window(window_type, N)
        
    def preprocess_audio(self, wav_file, sr=16000, n_fft=48, hop_length=256, n_mels=12):
        # Load audio using librosa (assumes WAV format)
        wav, _ = librosa.load(wav_file, sr=sr)

        # Frame division with overlap
        frames = librosa.util.frame(wav, frame_length=n_fft, hop_length=hop_length)

        # Apply window function (Hamming window by default)
        window = librosa.filters.get_window('hann', n_fft)
        windowed_frames = frames * window[:, np.newaxis]  # Reshape to have one column

        # FFT transformation
        fft_frames = np.fft.fft(windowed_frames, axis=1)
        
        # Compute MFCC features (using log-magnitude spectrum for stability)
        mfcc_features = librosa.feature.mfcc(y=np.abs(fft_frames), sr=sr, n_mels=n_mels, n_fft=n_fft)
        mfcc_features = np.mean(mfcc_features, axis=1)

        # Chroma_stft mean and variance
        chroma = librosa.feature.chroma_stft(y=wav, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_var = np.var(chroma, axis=1)

        # RMS mean and variance
        rms = librosa.feature.rms(y=wav)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)

        # Concatenate all features
        features = np.concatenate([mfcc_features, chroma_mean, chroma_var, [rms_mean], [rms_var]])

        return features

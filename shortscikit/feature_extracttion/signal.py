import numpy as np
import pywt
from scipy import signal
from sklearn.base import BaseEstimator,TransformerMixin


class SpectrogramTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, fft_length, stride_length, sample_rate, window="hanning", axis=0,
                 kw_args=None, inv_kw_args=None):
        self.sample_rate = sample_rate
        self.fft_length = fft_length
        self.stride_length = stride_length
        self.window = window
        self.axis = axis
        super().__init__()

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        _, _, Sxx = signal.spectrogram(x, fs=self.sample_rate, nperseg=self.fft_length,
                                       noverlap=self.fft_length - self.stride_length, window=self.window,
                                       axis=self.axis,
                                       return_onesided=True, mode="magnitude", scaling="density")
        return Sxx.transpose()


class WaeveletTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, wavelet_width, wavelet='mexh',
                 kw_args=None, inv_kw_args=None):
        self.wavelet_width = wavelet_width
        self.wavelet = wavelet
        super().__init__()

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        # TODO which is better? scipy.signal or pywt
        # wavelets = signal.cwt(X, signal.ricker, np.arange(1, self.wavelet_width + 1))
        wavelets, _ = pywt.cwt(x, np.arange(1, self.wavelet_width + 1), self.wavelet)
        return wavelets
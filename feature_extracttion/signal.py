import numpy as np
import pywt as pywt
from scipy import signal
from sklearn.base import BaseEstimator,TransformerMixin


class SpectrogramTransformer(BaseEstimator,TransformerMixin):

    def __init__(self, fft_length, stride_length, sample_rate, window="hanning", axis=0,
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        self.sample_rate = sample_rate
        self.fft_length = fft_length
        self.stride_length = stride_length
        self.window = window
        self.axis = axis

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        return self.f(x)

    def f(self, X):
        return self.to_spectrogram(X)

    def to_spectrogram(self, series):
        f, t, Sxx = signal.spectrogram(series, fs=self.sample_rate, nperseg=self.fft_length,
                                       noverlap=self.fft_length - self.stride_length, window=self.window,
                                       axis=self.axis,
                                       return_onesided=True, mode="magnitude", scaling="density")
        return Sxx.transpose()


class WaeveletTransformer(BaseEstimator,TransformerMixin):
    def __init__(self, wavelet_width, wavelet='mexh',
                 kw_args=None, inv_kw_args=None):
        validate = False
        inverse_func = None
        accept_sparse = False
        pass_y = 'deprecated'
        self.wavelet_width = wavelet_width
        self.wavelet = wavelet

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        return self.f(x)

    def f(self, X):
        # TODO which is better? scipy.signal or pywt
        # wavelets = signal.cwt(X, signal.ricker, np.arange(1, self.wavelet_width + 1))
        wavelets, _ = pywt.cwt(X, np.arange(1, self.wavelet_width + 1), self.wavelet)
        return wavelets

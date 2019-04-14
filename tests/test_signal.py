import numpy as np
from scipy import signal
import pywt
from nose2.tools  import such
from feature_extracttion.signal import SpectrogramTransformer,WaeveletTransformer
#>>> nose2 --plugin=nose2.plugins.layers
with such.A('sample test') as it:
    with it.having('SpectrogramTransformer test group'):
        # SpectrogramTransformer tests code refer to this site.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
        fs = 10e3
        N = 1e5
        amp = 2 * np.sqrt(2)
        noise_power = 0.01 * fs / 2
        time = np.arange(N) / float(fs)
        mod = 500*np.cos(2*np.pi*0.25*time)
        carrier = amp * np.sin(2*np.pi*3e3*time + mod)
        noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
        noise *= np.exp(-time/5)
        x = carrier + noise
        fft_length = 256
        stride_length = 224
        f, t, Sxx = signal.spectrogram(x, fs=fs, nperseg=fft_length,
                                       noverlap=fft_length - stride_length, window="hanning",
                                       axis=0, return_onesided=True, mode="magnitude", scaling="density")

        spectrogram_transformer = SpectrogramTransformer(fft_length, stride_length, fs, window="hanning", axis=0)

        @it.should('SpectrogramTransformer fit method')
        def spectrogram_test1(case):
            it.assertIsInstance(spectrogram_transformer.fit(x), SpectrogramTransformer)

        @it.should('SpectrogramTransformer transform method')
        def spectrogram_test2(case):
            it.assertTrue(np.alltrue(spectrogram_transformer.transform(x) == Sxx.transpose()))

        @it.should('SpectrogramTransformer fit_transform method')
        def spectrogram_test3(case):
            it.assertTrue(np.alltrue(spectrogram_transformer.fit_transform(x) == Sxx.transpose()))

    with it.having('WaeveletTransformer test group'):
        # WaeveletTransformer tests code refer to this site.
        # https://pywavelets.readthedocs.io/en/latest/ref/cwt.html?highlight=cwt
        
        t = np.linspace(-1, 1, 200, endpoint=False)
        sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
        widths = 30
        cwtmatr, freqs = pywt.cwt(sig, np.arange(1, widths + 1), 'mexh')
        waevelet_transformer = WaeveletTransformer(widths, 'mexh')
        
        @it.should('WaeveletTransformer fit method')
        def waevelet_test1(case):
            it.assertIsInstance(waevelet_transformer.fit(sig), WaeveletTransformer)

        @it.should('WaeveletTransformer transform method')
        def waevelet_test2(case):
            it.assertTrue(np.alltrue(waevelet_transformer.transform(sig) == cwtmatr))

        @it.should('WaeveletTransformer fit_transform method')
        def waevelet_test3(case):
            it.assertTrue(np.alltrue(waevelet_transformer.fit_transform(sig) == cwtmatr))

it.createTests(globals())
"""
Two feature representations computed per window, feeding the two branches
of EMGNetFusion (see model.py):

  1. CWT scalogram (Continuous Wavelet Transform, Mexican-hat wavelet) --
     the paper's own input representation (Section 3.1 / Figure 5).
  2. Classical hand-crafted time-domain (TD) features -- MAV/RMS/WL/VAR/
     ZC/SSC/WAMP/IEMG per channel. Added because a Random Forest trained on
     just these features scored well above the pure CWT-CNN on this
     dataset (see README) -- fusing them into the network lets EMGNet use
     that same signal instead of leaving it on the table.
"""
import numpy as np
import pywt

from . import config


def cwt_scalogram(signal_1d: np.ndarray) -> np.ndarray:
    """CWT of a single 1-D window -> (num_scales, len(signal_1d)) magnitude array."""
    coeffs, _freqs = pywt.cwt(
        signal_1d, scales=config.CWT_SCALES, wavelet=config.CWT_WAVELET
    )
    return np.abs(coeffs).astype(np.float32)


def _downsample_2d(arr: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool downsample by an integer factor along both axes."""
    if factor <= 1:
        return arr
    h, w = arr.shape
    h_trim, w_trim = h - (h % factor), w - (w % factor)
    arr = arr[:h_trim, :w_trim]
    return arr.reshape(h_trim // factor, factor, w_trim // factor, factor).mean(axis=(1, 3))


def window_to_cwt_tensor(window: np.ndarray) -> np.ndarray:
    """
    window: (num_channels, window_size) raw sensor values.
    returns: (num_channels, H, W) scalogram tensor.
    """
    maps = []
    for ch in range(window.shape[0]):
        scal = cwt_scalogram(window[ch])
        scal = _downsample_2d(scal, config.SPECTROGRAM_DOWNSAMPLE)
        maps.append(scal)
    return np.stack(maps, axis=0)


def _mav(x): return np.mean(np.abs(x))
def _rms(x): return np.sqrt(np.mean(x ** 2))
def _wl(x): return np.sum(np.abs(np.diff(x)))
def _var(x): return np.var(x)
def _zc(x): return np.sum((x[:-1] * x[1:]) < 0)
def _ssc(x): return np.sum((np.diff(x[:-1]) * np.diff(x[1:])) < 0)
def _wamp(x): return np.sum(np.abs(np.diff(x)) > config.WAMP_THRESHOLD)
def _iemg(x): return np.sum(np.abs(x))


def _power_spectrum(x):
    """rfft power spectrum + the Hz value of each bin, DC bin dropped (it's
    the window's mean, already captured by MAV/IEMG and not a frequency)."""
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / config.EFFECTIVE_SAMPLE_RATE_HZ)
    return freqs[1:], spectrum[1:]


def _mnf(x):
    """Mean (power-weighted average) frequency -- shifts with contraction
    type/muscle fatigue; standard EMG frequency-domain feature."""
    freqs, power = _power_spectrum(x)
    total = power.sum()
    return float(np.sum(freqs * power) / total) if total > 0 else 0.0


def _mdf(x):
    """Median frequency -- the frequency splitting total spectral power in half."""
    freqs, power = _power_spectrum(x)
    total = power.sum()
    if total <= 0:
        return 0.0
    cum = np.cumsum(power)
    idx = np.searchsorted(cum, total / 2.0)
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def _spectral_entropy(x):
    """Shannon entropy of the normalized power spectrum -- low when power
    concentrates in a narrow band, high when it's spread out."""
    _freqs, power = _power_spectrum(x)
    total = power.sum()
    if total <= 0:
        return 0.0
    p = power / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _peak_freq(x):
    """Frequency of the single largest spectral component."""
    freqs, power = _power_spectrum(x)
    if len(power) == 0:
        return 0.0
    return float(freqs[np.argmax(power)])


_TD_FUNCS = {"MAV": _mav, "RMS": _rms, "WL": _wl, "VAR": _var,
             "ZC": _zc, "SSC": _ssc, "WAMP": _wamp, "IEMG": _iemg,
             "MNF": _mnf, "MDF": _mdf, "SPEC_ENTROPY": _spectral_entropy,
             "PEAK_FREQ": _peak_freq}


def window_to_td_features(window: np.ndarray) -> np.ndarray:
    """
    window: (num_channels, window_size) raw sensor values.
    returns: (num_channels * len(config.TD_FEATURES),) float32 vector, channel-major
             (all of channel 0's features, then channel 1's, ...).
    """
    out = np.empty(window.shape[0] * len(config.TD_FEATURES), dtype=np.float32)
    i = 0
    for ch in range(window.shape[0]):
        x = window[ch]
        for name in config.TD_FEATURES:
            out[i] = _TD_FUNCS[name](x)
            i += 1
    return out


def cwt_output_spatial_shape() -> tuple:
    """(H, W) of window_to_cwt_tensor's output, for sizing the model / sanity checks."""
    dummy = np.random.randn(1, config.WINDOW_SIZE).astype(np.float32)
    out = window_to_cwt_tensor(dummy)
    return out.shape[1], out.shape[2]

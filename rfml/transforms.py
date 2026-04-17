"""Signal transforms for IQ data.

All transforms operate on 1-D complex numpy arrays (dtype complex64/128) unless
noted otherwise.  ComplexTo2D is always the last step and converts to float32
shape (2, N) expected by the downstream EfficientNet model.
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.ndimage import uniform_filter1d


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class RandomApply:
    def __init__(self, transform, probability):
        self.transform = transform
        self.probability = probability

    def __call__(self, data):
        if np.random.random() < self.probability:
            return self.transform(data)
        return data


class Normalize:
    def __init__(self, norm=2):
        self.norm = norm

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if self.norm == np.inf:
            peak = np.max(np.abs(data))
            if peak > 0:
                data = data / peak
        elif self.norm == 2:
            rms = np.sqrt(np.mean(np.abs(data) ** 2))
            if rms > 0:
                data = data / rms
        else:
            n = np.linalg.norm(data.ravel(), ord=self.norm)
            if n > 0:
                data = data / n
        return data


class ComplexTo2D:
    """Convert a complex (N,) array to a float32 (2, N) array [I; Q]."""

    def __call__(self, data: np.ndarray) -> np.ndarray:
        return np.stack([data.real, data.imag], axis=0).astype(np.float32)


class RandomPhaseShift:
    def __init__(self, phase_offset=(-1, 1)):
        self.phase_offset = phase_offset

    def __call__(self, data: np.ndarray) -> np.ndarray:
        phase = np.random.uniform(self.phase_offset[0], self.phase_offset[1]) * np.pi
        return data * np.exp(1j * phase)


class RandomTimeShift:
    def __init__(self, time_shift=(-10, 10)):
        self.time_shift = time_shift

    def __call__(self, data: np.ndarray) -> np.ndarray:
        shift = np.random.randint(self.time_shift[0], self.time_shift[1] + 1)
        return np.roll(data, shift)


class RandomFrequencyShift:
    def __init__(self, freq_shift=(-0.1, 0.1)):
        self.freq_shift = freq_shift

    def __call__(self, data: np.ndarray) -> np.ndarray:
        shift = np.random.uniform(self.freq_shift[0], self.freq_shift[1])
        t = np.arange(len(data))
        return data * np.exp(2j * np.pi * shift * t)


class RandomResample:
    def __init__(self, rate_ratio=(0.75, 1.5), num_iq_samples=None):
        self.rate_ratio = rate_ratio
        self.num_iq_samples = num_iq_samples

    def __call__(self, data: np.ndarray) -> np.ndarray:
        ratio = np.random.uniform(self.rate_ratio[0], self.rate_ratio[1])
        new_len = max(1, int(len(data) * ratio))
        data = sp_signal.resample(data, new_len)
        if self.num_iq_samples is not None:
            if len(data) >= self.num_iq_samples:
                data = data[: self.num_iq_samples]
            else:
                data = np.pad(data, (0, self.num_iq_samples - len(data)))
        return data


class RayleighFadingChannel:
    def __init__(self, coherence_bandwidth=(0.05, 0.5), power_delay_profile=(1.0,)):
        self.coherence_bandwidth = coherence_bandwidth
        self.power_delay_profile = np.asarray(power_delay_profile, dtype=float)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        n = len(data)
        coh_bw = np.random.uniform(
            self.coherence_bandwidth[0], self.coherence_bandwidth[1]
        )
        pdp = self.power_delay_profile / self.power_delay_profile.sum()
        smoothing = max(1, int(round(1.0 / coh_bw)))
        output = np.zeros(n, dtype=complex)
        for delay, power in enumerate(pdp):
            h_i = uniform_filter1d(np.random.randn(n), size=smoothing)
            h_q = uniform_filter1d(np.random.randn(n), size=smoothing)
            h = np.sqrt(power / 2.0) * (h_i + 1j * h_q)
            output += h * np.roll(data, delay)
        return output


class IQImbalance:
    def __init__(
        self,
        amplitude_imbalance=(-3, 3),
        phase_imbalance=(-0.1, 0.1),
        dc_offset=(-0.1, 0.1),
    ):
        self.amplitude_imbalance = amplitude_imbalance
        self.phase_imbalance = phase_imbalance
        self.dc_offset = dc_offset

    def __call__(self, data: np.ndarray) -> np.ndarray:
        amp_db = np.random.uniform(
            self.amplitude_imbalance[0], self.amplitude_imbalance[1]
        )
        amp = 10 ** (amp_db / 20.0)
        phase = np.random.uniform(self.phase_imbalance[0], self.phase_imbalance[1])
        dc_i = np.random.uniform(self.dc_offset[0], self.dc_offset[1])
        dc_q = np.random.uniform(self.dc_offset[0], self.dc_offset[1])
        i, q = data.real, data.imag
        i_out = amp * (i * np.cos(phase) - q * np.sin(phase)) + dc_i
        q_out = i * np.sin(phase) + q * np.cos(phase) + dc_q
        return i_out + 1j * q_out


class TargetSNR:
    def __init__(self, snr_db=(-2, 30), eb_no=False):
        self.snr_db = snr_db
        self.eb_no = eb_no

    def __call__(self, data: np.ndarray) -> np.ndarray:
        snr = np.random.uniform(self.snr_db[0], self.snr_db[1])
        snr_linear = 10 ** (snr / 10.0)
        signal_power = np.mean(np.abs(data) ** 2)
        noise_power = signal_power / snr_linear if snr_linear > 0 else 0.0
        noise = np.sqrt(noise_power / 2.0) * (
            np.random.randn(len(data)) + 1j * np.random.randn(len(data))
        )
        return data + noise

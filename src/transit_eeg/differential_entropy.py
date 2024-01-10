import numpy as np
from scipy.signal import butter, lfilter

class ButterBandpassFilter:
    def __init__(self, low_cut, high_cut, fs, order=5):
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.fs = fs
        self.order = order

    def apply(self, signal):
        nyq = 0.5 * self.fs
        low = self.low_cut / nyq
        high = self.high_cut / nyq
        b, a = butter(self.order, [low, high], btype='band')
        return lfilter(b, a, signal)


class BandSignalSplitter:
    def __init__(self, sampling_rate=128, order=5, band_dict=None):
        self.sampling_rate = sampling_rate
        self.order = order
        self.band_dict = band_dict or {
            "theta": [4, 8],
            "alpha": [8, 14],
            "beta": [14, 31],
            "gamma": [31, 49]
        }

    def apply(self, eeg):
        band_list = []
        for low, high in self.band_dict.values():
            b, a = butter(self.order, [low, high], fs=self.sampling_rate, btype="band")
            band_list.append(lfilter(b, a, eeg))
        return np.stack(band_list, axis=0)


class BandDifferentialEntropyCalculator:
    def apply(self, eeg):
        return 1 / 2 * np.log2(2 * np.pi * np.e * np.var(eeg))


class EEGGridProjector:
    def __init__(self, channel_location_dict):
        self.channel_location_dict = channel_location_dict

    def apply(self, eeg):
        num_electrodes = len(self.channel_location_dict)
        outputs = np.zeros([num_electrodes, eeg.shape[1]])

        for i, (x, y) in enumerate(self.channel_location_dict.values()):
            outputs[i] = eeg[x][y]

        return outputs


if __name__ == '__main__':
    from transit_eeg.constants import format_channel_location_dict, SEED_CHANNEL_LIST, CHANNEL_LOCATION_10_20, PHYAAT_CHANNEL_LIST
    CHANNEL_LOCATION_DICT = format_channel_location_dict(
        PHYAAT_CHANNEL_LIST, CHANNEL_LOCATION_10_20
    )
    print(CHANNEL_LOCATION_DICT)
    fs = 128
    eeg_data = np.random.randn(32, 128)
    bandpass_filter = ButterBandpassFilter(low_cut=4, high_cut=40, fs=fs)
    band_signal_splitter = BandSignalSplitter(sampling_rate=fs)
    entropy_calculator = BandDifferentialEntropyCalculator()
    grid_projector = EEGGridProjector(channel_location_dict=CHANNEL_LOCATION_DICT) 

    filtered_eeg = bandpass_filter.apply(eeg_data)
    print(filtered_eeg.shape)
    split_bands = band_signal_splitter.apply(filtered_eeg)
    print(split_bands.shape)
    differential_entropy = entropy_calculator.apply(split_bands)
    print(differential_entropy)

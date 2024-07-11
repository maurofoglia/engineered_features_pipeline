import numpy as np
import torcheeg.transforms as transforms


class Features:

    def __init__(self):



        # Band name and the critical sampling rate or frequencies
        band_dict = {'delta': (1, 4),
                     'theta': (4, 8),
                     'alpha': (8, 14),
                     'beta': (14, 30)}

        # The sampling period for the frequencies output in Hz
        sampling_rate = 256

        # A transform method to convert EEG signals of each channel into spectrograms using wavelet transform
        # Returns the spectrograms based on the wavelet transform for all electrodes
        # np.ndarray[number of electrodes, total_scale, number of data points]
        self.cwt = transforms.CWTSpectrum(sampling_rate=sampling_rate)

        # A transform method to split the EEG signal into signals in different sub-bands
        # Returns the differential entropy of several sub-bands for all electrodes
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_signal = transforms.BandSignal(sampling_rate=sampling_rate, band_dict=band_dict)

        # A transform method for calculating the differential entropy of EEG signals in several sub-bands with EEG signals as input
        # The differential entropy of several sub-bands for all electrodes
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_diff_entropy = transforms.BandDifferentialEntropy(sampling_rate=sampling_rate, band_dict=band_dict)

        # A transform method for calculating the power spectral density of EEG signals in several sub-bands with EEG signals as input.
        # Returns the power spectral density of several sub-bands for all electrodes.
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_pow_spect_density = transforms.BandPowerSpectralDensity(sampling_rate=sampling_rate, band_dict=band_dict)

        # A transform method for calculating the mean absolute deviation of EEG signals in several sub-bands with EEG signals as input.
        # Returns the mean absolute deviation of several sub-bands for all electrodes.
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_mad = transforms.BandMeanAbsoluteDeviation(sampling_rate=sampling_rate, band_dict=band_dict)

        # A transform method for calculating the kurtosis of EEG signals in several sub-bands with EEG signals as input.
        # Returns the kurtosis of several sub-bands for all electrodes.
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_kurt = transforms.BandKurtosis(sampling_rate=sampling_rate, band_dict=band_dict)

        # A transform method for calculating the skewness of EEG signals in several sub-bands with EEG signals as input.
        # Returns the skewness of several sub-bands for all electrodes.
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_skew = transforms.BandSkewness(sampling_rate=sampling_rate, band_dict=band_dict)

        # Splitting the EEG signal from each electrode into two functions using wavelet decomposition.
        # Returns EEG signal after wavelet decomposition, where 2 corresponds to the two functions of the wavelet decomposition, and number of data points / 2 represents the length of each component
        # np.ndarray[number of electrodes, 2, number of data points / 2]
        self.dwt = transforms.DWTDecomposition()

        # A transform method for calculating the approximate entropy of EEG signals in several sub-bands with EEG signals as input
        # Returns the differential entropy of several sub-bands for all electrodes.
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_appr_entropy = transforms.BandApproximateEntropy(sampling_rate=sampling_rate, band_dict=band_dict)

        # A transform method for calculating the approximate entropy of EEG signals in several sub-bands with EEG signals as input
        # Returns the differential entropy of several sub-bands for all electrodes.
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_sample_entropy = transforms.BandSampleEntropy(sampling_rate=sampling_rate, band_dict=band_dict)

        # A transform method for calculating the SVD entropy of EEG signals in several sub-bands with EEG signals as input
        # Returns the differential entropy of several sub-bands for all electrodes.
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_svd = transforms.BandSVDEntropy(sampling_rate=sampling_rate, band_dict=band_dict, Tau=2)

        # A transform method for calculating the detrended fluctuation analysis (DFA) of EEG signals in several sub-bands with EEG signals as input
        # Returns the differential entropy of several sub-bands for all electrodes.
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_dfa = transforms.BandDetrendedFluctuationAnalysis(sampling_rate=sampling_rate, band_dict=band_dict)

        # A transform method for calculating the petrosian fractal dimension (PFD) of EEG signals in several sub-bands with EEG signals as input
        # Returns The differential entropy of several sub-bands for all electrodes.
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_pfd = transforms.BandPetrosianFractalDimension(sampling_rate=sampling_rate, band_dict=band_dict)

        # A transform method for calculating the higuchi fractal dimension (HFD) of EEG signals in several sub-bands with EEG signals as input
        # Returns The differential entropy of several sub-bands for all electrodes.
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_hfd = transforms.BandHiguchiFractalDimension(sampling_rate=sampling_rate, band_dict=band_dict)

        # A transform method for calculating the hjorth mobility/complexity of EEG signals in several sub-bands with EEG signals as input
        # Returns The differential entropy of several sub-bands for all electrodes.
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_hjorth = transforms.BandHjorth(sampling_rate=sampling_rate, band_dict=band_dict)

        # A transform method for calculating the hurst exponent of EEG signals in several sub-bands with EEG signals as input
        # Returns The differential entropy of several sub-bands for all electrodes.
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_hurst = transforms.BandHurst(sampling_rate=sampling_rate, band_dict=band_dict)

        # A transform method for calculating the power of EEG signals in several sub-bands with EEG signals as input
        # Returns The differential entropy of several sub-bands for all electrodes.
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_bin_pow = transforms.BandBinPower(sampling_rate=sampling_rate, band_dict=band_dict)

        # Calculate autoregression reflection coefficients on the input data
        # Returns The autoregression reflection coefficients
        # np.ndarray [number of electrodes, order]
        self.arr = transforms.ARRCoefficient()

        # A transform method for calculating the spectral entropy of EEG signals in several sub-bands with EEG signals as input
        # Returns the differential entropy of several sub-bands for all electrodes
        # np.ndarray[number of electrodes, number of sub-bands]
        self.b_spect_entropy = transforms.BandSpectralEntropy(sampling_rate=sampling_rate, band_dict=band_dict)

        # A transform method to calculate the phase locking values between the EEG signals of different electrodes
        # Returns the phase locking values between EEG signals of different electrodes
        # np.ndarray[number of electrodes, number of electrodes]
        self.plc = transforms.PhaseLockingCorrelation()

    # metodo che accetta come parametri di ingresso, oltre alle proprietà della classe, epoca, dizionario da popolare e il trial corrente
    # restituisce il dizionario popolato con i dati delle features applicate
    def transform(self, epoch, dict, trial):

        # electrodes list
        ch_names = ['AF3', 'TP9', 'TP10', 'AF4']

        for feature_name in dict.keys():
            if feature_name == 'cwt':
                features_data = self.cwt(eeg=epoch)
            elif feature_name == 'b_signal':
                features_data = self.b_signal(eeg=epoch)
            elif feature_name == 'b_diff_entropy':
                features_data = self.b_diff_entropy(eeg=epoch)
            elif feature_name == 'b_pow_spect_density':
                features_data = self.b_pow_spect_density(eeg=epoch)
            elif feature_name == 'b_mad':
                features_data = self.b_mad(eeg=epoch)
            elif feature_name == 'b_kurt':
                features_data = self.b_kurt(eeg=epoch)
            elif feature_name == 'b_skew':
                features_data = self.b_skew(eeg=epoch)
            elif feature_name == 'dwt':
                dwt = self.dwt(eeg=epoch)
                features_data = {'eeg': []}
                # features_data = dwt['eeg'] = np.hstack((dwt['eeg'][0:1,:,:].squeeze(), dwt['eeg'][1:2,:,:].squeeze()))
                features_data['eeg'] = np.hstack((dwt['eeg'][0:1, :, :].squeeze(), dwt['eeg'][1:2, :, :].squeeze()))
            elif feature_name == 'b_appr_entropy':
                features_data = self.b_appr_entropy(eeg=epoch)
            elif feature_name == 'b_sample_entropy':
                features_data = self.b_sample_entropy(eeg=epoch)
            elif feature_name == 'b_svd':
                features_data = self.b_svd(eeg=epoch)
            elif feature_name == 'b_dfa':
                features_data = self.b_dfa(eeg=epoch)
            elif feature_name == 'b_pfd':
                features_data = self.b_pfd(eeg=epoch)
            elif feature_name == 'b_hfd':
                features_data = self.b_hfd(eeg=epoch)
            elif feature_name == 'b_hjorth':
                features_data = self.b_hjorth(eeg=epoch)
            elif feature_name == 'b_hurst':
                features_data = self.b_hurst(eeg=epoch)
            elif feature_name == 'b_bin_pow':
                features_data = self.b_bin_pow(eeg=epoch)
            elif feature_name == 'arr':
                features_data = self.arr(eeg=epoch)
            elif feature_name == 'b_spect_entropy':
                features_data = self.b_spect_entropy(eeg=epoch)
            elif feature_name == 'plc':
                plc = self.plc(eeg=epoch)
                plc['eeg'] = plc['eeg'].squeeze()
                features_data = plc


            electrodes = {}
            # Usa un ciclo for per separare l'array originale
            eeg_data = features_data['eeg']

            for ch in ch_names:
                for i in range(4):
                    # electrodes[ch] = features_data[i]
                    electrodes[ch] = eeg_data[i]
            # inserimento dei dati della feature corrispondente alla keyword corrente
            dict[feature_name].append([trial, electrodes])

        return dict

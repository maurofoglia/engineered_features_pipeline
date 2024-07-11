import features
import eeg_utils_2
import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt

import SPA_EEG

plt.show()

with open('dataframe_0.pkl', 'rb') as handle:
    dataframe = pkl.load(handle)

l_freq = 1
h_freq = 30
threshold = 50
for name_ in dataframe.keys():  # iterazione per ogni nome (chiave) del dizionario dataframe
    # name_ = 'Paola_Carbone'
    Epochs = []  # elenco vuoto
    target = []
    Y = []
    raw = []
    print(name_)
    for k in range(40):  # si itera per i primi 40 elementi dei dati relativi all'utente corrente
        try:
            # 2. generate mne raw
            raw = eeg_utils_2.generate_raw(dataframe[name_][k])

            epochs = eeg_utils_2.create_epochs(raw, t_min=0,
                                               t_max=5)  # creazione epoch limitando a un intervallo di tempo
            epochs_video = epochs['start_video_marker']
            # SPA
            epochs_video = eeg_utils_2.filter_raw(epochs_video, l_freq=l_freq, h_freq=h_freq)
            epochs_video = SPA_EEG.SPA(epochs_video)

            # ICA stimuli
            # epochs_video = eeg_utils_2.preprocessing_eeg(epochs_video, l_freq=1, h_freq=100, prob_threshold=0.8)
            # BPF stimuli
            # epochs_video = eeg_utils_2.filter_raw(epochs_video, l_freq=l_freq, h_freq=h_freq)

            # trial
            trial = dataframe[name_][k]['session_id']
            trial = int(trial.split('S')[1])

            # epochs_fixation_cross_marker
            epochs_baseline = epochs['fixation_cross_marker']

            # SPA baseline
            epochs_baseline = eeg_utils_2.filter_raw(epochs_baseline, l_freq=l_freq, h_freq=h_freq)
            epochs_baseline = SPA_EEG.SPA(epochs_baseline)


            # epochs_video.compute_psd().plot()


            print("")

            #ICA stimuli
            epochs_baseline = eeg_utils_2.preprocessing_eeg(epochs_baseline, l_freq=1, h_freq=100, prob_threshold=0.8)
            #BPF stimuli
            epochs_baseline = eeg_utils_2.filter_raw(epochs_baseline, l_freq=l_freq, h_freq=h_freq)

            band_dict = {'delta': (1, 4),
                         'theta': (4, 8),
                         'alpha': (8, 14),
                         'beta': (14, 30)}

            Epochs.append([epochs_video, trial, epochs_baseline])


            np.savez('epochs_data_with_channels.pz', data=epochs_video.get_data(), channels=epochs_video.ch_names)
            arrs = []
            epochs_video1 = epochs_video.get_data(copy=True).squeeze()

            #dict = {ch: epochs_video1[ch, :] for ch in epochs_video.ch_names}
            ch_names = ['AF3', 'TP9', 'TP10', 'AF4']
            dict = {}
            for ch in ch_names:
                for i in range(4):
                    dict[ch] = epochs_video1[i, :]

            print("")
            # Band name and the critical sampling rate or frequencies
            band_dict = {'delta': (1, 4),
                         'theta': (4, 8),
                         'alpha': (8, 14),
                         'beta': (14, 30)}

            # The sampling period for the frequencies output in Hz
            sampling_rate = 256
            arr = dict['AF3']
            #import features
            apply = features.Features()
            import torcheeg.transforms as transforms
            t = transforms.BandPowerSpectralDensity(sampling_rate=sampling_rate, band_dict=band_dict)
            t2 = epochs_video.get_data().squeeze().BandPowerSpectralDensity(sampling_rate=sampling_rate, band_dict=band_dict)
        except:
            print('')

    directory = 'epochs_preprocessed_0/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    path_id = directory + name_ + '.pkl'
    pkl.dump(Epochs, open(path_id, 'wb'))

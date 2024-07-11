import mne
from mne.filter import filter_data
from mne_icalabel import label_components
import numpy as np
import pandas as pd
from mne.preprocessing import ICA, EOGRegression, Xdawn


def calculate_markers(marker_1, marker_2):
    fixation_cross_marker = marker_1
    start_video_marker = marker_2
    return fixation_cross_marker, start_video_marker


def generate_raw(data):
    # create raw
    eeg = data['time_series']
    eeg = np.delete(eeg, 4, axis=1).T
    dataEEG = eeg * 1e-6
    ch_names = ['AF3', 'TP9', 'TP10', 'AF4']
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg']
    sfreq = 256.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(dataEEG, info)
    raw.set_montage('standard_1020')
    # raw.plot()
    # add markers
    onset = [data['fixation_cross_marker'],
             data['start_video_marker']]  # The starting time of annotations in seconds after orig_time
    duration = 0.01  # Durations of the annotations in seconds. If a float, all the annotations are given the same duration.
    marker = ['fixation_cross_marker', 'start_video_marker']
    annot_new = mne.Annotations(onset=onset,
                                duration=duration,
                                description=marker)
    raw.set_annotations(annot_new)
    return raw


def create_epochs(raw, t_min, t_max):
    events, event_dict = mne.events_from_annotations(raw)
    event_mapping = {'fixation_cross_marker': 1, 'start_video_marker': 2}
    baseline = (None, 0)
    epochs = mne.Epochs(raw, events, event_mapping, t_min, t_max, baseline=None, preload=True)
    return epochs


def preprocessing_eeg(epochs, l_freq, h_freq, prob_threshold):
    epochs_ica = epochs.copy()
    epochs_ica.filter(1, 100)
    epochs.set_eeg_reference('average')
    ica = ICA(n_components=0.99,
              max_iter="auto",
              method="infomax",
              random_state=97,
              fit_params=dict(extended=True),
              verbose=1
              )
    ica.fit(epochs_ica)
    ic_labels = label_components(epochs_ica, ica, method="iclabel")
    labels = ic_labels["labels"]
    data_prob = pd.DataFrame(ic_labels)
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    data_prob = data_prob.loc[exclude_idx]
    rslt_df = data_prob.loc[data_prob['y_pred_proba'] > prob_threshold]
    exclude_idx = rslt_df.index
    ica.apply(epochs, exclude=exclude_idx)
    return epochs


def create_epochs_scratch(eeg_transformed) -> object:
    ch_names = ['AF3', 'TP9', 'TP10', 'AF4']
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg']
    sfreq = 256.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    simulated_epochs = mne.EpochsArray(eeg_transformed, info)
    simulated_epochs.set_montage('standard_1020')
    return simulated_epochs


def filter_raw(raw_data, l_freq, h_freq):
    filtered_data = raw_data.copy()
    filtered_data = raw_data.filter(l_freq, h_freq)
    return filtered_data

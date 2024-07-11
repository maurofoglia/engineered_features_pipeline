import os
import pickle as pkl
import features
import pandas as pd
import numpy as np
# directory target inerente alle features, contiene una sottocartella per ciascun utente
directory = 'features_0/'
if not os.path.exists(directory):
    os.makedirs(directory)

# lista dei nomi delle features selezionate
features_list = ['cwt', 'b_signal', 'b_diff_entropy', 'b_pow_spect_density',
                 'b_mad', 'b_kurt', 'b_skew',
                 'dwt', 'b_appr_entropy', 'b_sample_entropy',
                 'b_svd', 'b_dfa', 'b_pfd',
                 'b_hfd', 'b_hjorth', 'b_hurst',
                 'b_bin_pow', 'arr', 'b_spect_entropy', 'plc']

# directory sorgente delle epoche
epochs_path = 'epochs_preprocessed_0/'

for name in os.listdir(epochs_path):
    # name = 'utente.pkl'
    user_name = name.split('.')[0]
    # user_name = 'utente'
    print(user_name)

    # creazione sottocartelle utente
    user_path = directory + user_name + '/'
    # user_path = 'features_0/utente/'
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    # creazione dei dizionari per le epoche relative a stimuli e baseline
    stimuli_dict = {feature: [] for feature in features_list}
    baseline_dict = {feature: [] for feature in features_list}

    name_path = os.path.join(epochs_path, name)
    # name_path = 'epochs_preprocessed_0/utente.pkl'

    # apertura dataframe
    with open(name_path, 'rb') as r:
        dataframe = pkl.load(r)

    #ch_names = ['AF3', 'TP9', 'TP10', 'AF4']

    for trial in dataframe:
        # numero del trial
        i = trial[1]
        print('Trial : ' + str(i))

        # epoche stimuli e baseline associate al trial corrente
        stimuli_epoch = trial[0].get_data().squeeze() * 1e6  # in microVolt
        baseline_epoch = trial[2].get_data().squeeze() * 1e6  # in microVolt

        # stimuli_epoch_dict = {ch_names[i]: stimuli_epoch[i, :] for i in range(len(ch_names))}
        # baseline_epoch_dict = {ch_names[i]: stimuli_epoch[i, :] for i in range(len(ch_names))}

        # ch_names = trial[0].info['ch_names']
        #
        # stimuli_epoch = pd.DataFrame(stimuli_epoch, index=ch_names)
        # baseline_epoch = pd.DataFrame(baseline_epoch, index=ch_names)
        # stimuli_epoch = np.column_stack([stimuli_epoch.index.values, stimuli_epoch.values])
        # baseline_epoch = np.column_stack([baseline_epoch.index.values, baseline_epoch.values])

        # istanziamento dell'oggetto Features
        apply = features.Features()

        # applicazione del metodo transform della classe Features alle epoche
        stimuli_dict = apply.transform(stimuli_epoch, stimuli_dict, i)
        baseline_dict = apply.transform(baseline_epoch, baseline_dict, i)

    # path destinazione dei dizionari
    stimuli_path = os.path.join(user_path, user_name + '_stimuli' + '.pkl')
    baseline_path = os.path.join(user_path, user_name + '_baseline' + '.pkl')

    # dump dei dizionari nel path corrispondente
    pkl.dump(stimuli_dict, open(stimuli_path, 'wb'))
    pkl.dump(baseline_dict, open(baseline_path, 'wb'))

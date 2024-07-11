import glob
import pickle as pkl
import numpy as np
import os
import pandas as pd


def features_list(subject_files):
    for file_path in subject_files['subj_file']:
        # Carica i dati per stimuli e baseline
        print(file_path)
        path = file_path + '/'
        data_files = glob.glob(path + '/*pkl')
        data_stimuli = pkl.load(open(data_files[1], 'rb'))
        features_list = list(data_stimuli.keys())
    return features_list


def process_subject_data_X(subject_files, features_list, class_indices):
    ch_names = ['AF3', 'TP9', 'TP10', 'AF4']
    X = pd.DataFrame()

    for file_path in subject_files['subj_file']:
        path = file_path + '/'
        data_files = glob.glob(path + '/*pkl')
        data_stimuli = pkl.load(open(data_files[1], 'rb'))
        data_baseline = pkl.load(open(data_files[0], 'rb'))
        user_name = os.path.basename(file_path)
        print(user_name)

        for feature in features_list:
            print(feature)
            # Estrai e processa la feature specificata per gli stimoli
            feature_stimuli = data_stimuli[feature]
            #processed_stimuli = process_feature(feature_stimuli, class_indices)
            #y_stimuli = np.zeros(len(processed_stimuli['AF3'][0])) + 1
            # Estrai e processa la feature specificata per il baseline
            feature_baseline = data_baseline[feature]
            processed_baseline = process_feature(feature_baseline, class_indices)
            #y_baseline = np.zeros(len(processed_baseline['AF3'][0]))
            #y = np.concatenate((y_stimuli, y_baseline))

            for trial_index in class_indices:

                try:
                    feature_class_stimuli = feature_stimuli[trial_index]

                except IndexError:
                    print(f"Trial index {trial_index} not found in stimuli data for {user_name}")
                    break

                try:
                    feature_class_baseline = feature_baseline[trial_index]

                except IndexError:
                    print(f"Trial index {trial_index} not found in baseline data for {user_name}")
                    break

                for ch in ch_names:
                    column_label = f"{ch}_{feature}"

                    row_data_stimuli = np.array(feature_class_stimuli[1][ch]).flatten()
                    row_label_stimuli = f"{user_name}_trial{trial_index}_stimuli"
                    for i, value in enumerate(row_data_stimuli):
                        column_label_stimuli = f"{column_label}_{i}"
                        X.loc[row_label_stimuli, column_label_stimuli] = value


                    row_data_baseline = np.array(feature_class_baseline[1][ch]).flatten()
                    row_label_baseline = f"{user_name}_trial{trial_index}_baseline"
                    for j, value in enumerate(row_data_baseline):
                        column_label_baseline = f"{column_label}_{j}"
                        X.loc[row_label_baseline, column_label_baseline] = value

    row_labels = X.index.tolist()

    # Crea una nuova lista di etichette di riga alternate tra '0' e '1'
    new_row_labels = ['1' if i % 2 == 0 else '0' for i in range(len(row_labels))]

    # Crea un dizionario per mappare le vecchie etichette alle nuove etichette alternate
    rename_mapping = dict(zip(row_labels, new_row_labels))

    # Rinomina le etichette di riga nel DataFrame utilizzando il dizionario di mapping
    X = X.rename(index=rename_mapping)

    return X



def process_subject_data_y(subject_files, features_list, class_indices):
    ch_names = ['AF3', 'TP9', 'TP10', 'AF4']
    y_stimuli = []
    y_baseline = []
    y = []
    y_stimuli = np.empty_like(y_stimuli)
    y_baseline = np.empty_like(y_baseline)
    y = np.empty_like(y)
    for file_path in subject_files['subj_file']:
        path = file_path + '/'
        data_files = glob.glob(path + '/*pkl')
        data_stimuli = pkl.load(open(data_files[1], 'rb'))
        data_baseline = pkl.load(open(data_files[0], 'rb'))
        user_name = os.path.basename(file_path)
        feature = features_list[0]
        feature_stimuli = data_stimuli[feature]
        feature_baseline = data_baseline[feature]
        for trial_index in class_indices:

            try:
                processed_stimuli = process_feature(feature_stimuli, class_indices)
                y = np.append(y, np.zeros(len(processed_stimuli['AF3'][0])) + 1)
            except:
                print(f"Trial index {trial_index} not found in stimuli data for {user_name}")
                continue

            try:
                processed_baseline = process_feature(feature_baseline, class_indices)
                y = np.append(y, np.zeros(len(processed_baseline['AF3'][0])))
            except:
                print(f"Trial index {trial_index} not found in baseline data for {user_name}")
                continue


    return y


def process_feature(feature_data, class_indices):
    try:
        feature_class = [feature_data[i] for i in class_indices]
        # processed_feature = [feature_class[k][1]['eeg'].flatten() for k in range(len(feature_class))]
        ch_names = ['AF3', 'TP9', 'TP10', 'AF4']
        processed_feature = {ch: [] for ch in ch_names}
        for ch in ch_names:
            processed_data = [feature_class[k][1][ch].flatten() for k in range(len(feature_class))]
            processed_feature[ch].append(processed_data)


    except:
        processed_feature = []

    return processed_feature

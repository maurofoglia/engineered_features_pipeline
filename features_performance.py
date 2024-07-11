import os
import pickle
import pandas as pd
import re

# Path main directory
main_dir = 'quadranti'

# file di output Excel dove saranno esportati i dati
output_file = 'output_features_performance.xlsx'

# Crea un dizionario per memorizzare i DataFrame di ciascun quadrante
quadranti = {}

# Iterazione attraverso le cartelle quadrante_1, quadrante_2, ecc...
for quadrante in os.listdir(main_dir):
    quadrante_path = os.path.join(main_dir, quadrante)
    quadrante_data = {}

    # Iterazione attraverso i file features.pkl nella cartella corrente
    for feature in os.listdir(quadrante_path):
        file_path = os.path.join(quadrante_path, feature)
        feature_name = feature.split('.')[0].split('_quadrante')[0]
        print(f'Loading feature: {feature_name} per {quadrante} ')

        # Caricamento file .pkl
        with open(file_path, 'rb') as f:
            dataframe = pickle.load(f)
            for i, user in enumerate(dataframe, start=1):
                print('Utente ' + str(i) + ': ' + str(user))
                print(dataframe[user][1])
                print(dataframe[user][2])

                confusion_matrix = dataframe[user][1]
                true_positive = confusion_matrix[0, 0]
                false_positive = confusion_matrix[0, 1]
                false_negative = confusion_matrix[1, 0]
                true_negative = confusion_matrix[1, 1]

                classification_report = dataframe[user][2]
                print('')

                label0_precision = classification_report['0']['precision']
                label0_recall = classification_report['0']['recall']
                label0_f1score = classification_report['0']['f1-score']
                label1_precision = classification_report['1']['precision']
                label1_recall = classification_report['1']['recall']
                label1_f1score = classification_report['1']['f1-score']
                accuracy = classification_report['accuracy']

                # valori estratti aggiunti al dict dei dati del quadrante
                if user not in quadrante_data:
                    quadrante_data[user] = {}

                quadrante_data[user][f'{feature_name}_true_positive'] = true_positive
                quadrante_data[user][f'{feature_name}_false_positive'] = false_positive
                quadrante_data[user][f'{feature_name}_false_negative'] = false_negative
                quadrante_data[user][f'{feature_name}_true_negative'] = true_negative
                quadrante_data[user][f'{feature_name}_precision_0'] = label0_precision
                quadrante_data[user][f'{feature_name}_recall_0'] = label0_recall
                quadrante_data[user][f'{feature_name}_f1score_0'] = label0_f1score
                quadrante_data[user][f'{feature_name}_precision_1'] = label1_precision
                quadrante_data[user][f'{feature_name}_recall_1'] = label1_recall
                quadrante_data[user][f'{feature_name}_f1score_1'] = label1_f1score
                quadrante_data[user][f'{feature_name}_accuracy'] = accuracy


    # Creazione del DataFrame per il quadrante corrente
    df = pd.DataFrame.from_dict(quadrante_data, orient='index')
    quadranti[quadrante] = df

# Esporta i DataFrame in un file Excel
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for quadrante, df in quadranti.items():
        # Scrittura del DataFrame nel foglio Excel
        df.to_excel(writer, sheet_name=quadrante, index=True)

print(f'Export completato in: {output_file}')
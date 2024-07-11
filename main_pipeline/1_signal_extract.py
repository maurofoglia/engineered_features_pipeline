from glob import glob
import mne
import numpy as np
from pyxdf import pyxdf
import pickle as pkl
import user_session_utils
import eeg_utils_2

users_data = glob('Dati_Utenti/*')
dataframe = {}  # dizionario vuoto da popolare con i dati delle sessioni utente
corrupted_data = []  # lista per tenere traccia dei dati non validi
for user_file in users_data:  # itero per ogni file utente
    print(user_file)
    user_session = glob(user_file + '/*')
    user_session.sort()
    for session in user_session:
        print(session)
        path = session + '/eeg/'
        pat_session = glob(path + '/*')
        try:
            data, header = pyxdf.load_xdf(pat_session[0])  # carica i dati e l'intestazione dal primofile trovato
            data = user_session_utils.correct_session(data)  # corregge sessione utente
            markers = eeg_utils_2.calculate_markers(marker_1=0, marker_2=8)  # calcola i marcatori
            user_name = user_file.split('/')  # suddivide i file utente
            user_name = user_name[0].split('-')  # suddivide i nomi utente
            session_number = session.split('/')  # suddivide i numeri di sessione
            session_number = session_number[0].split('\\')
            session_number = session_number[2].split('-')

            session_data = {"session_id": session_number[1], "time_series": data[1]['time_series'],
                       "fixation_cross_marker": markers[0], "start_video_marker": markers[1]}  # dizionario con informazione della sessione corrente
            # create dataframe # dizionario con chiavi (nomi utenti) e valori (liste di dati delle sessioni)
            if user_name[1] in dataframe:
                dataframe[user_name[1]].append(session_data) # aggiorna dizionario aggiungendo la sessione al nome utente
            else:
                dataframe.update({user_name[1]: [session_data]})

        except:
            corrupted_data.append(pat_session[0])  # eccezione: percorso del file problematico aggiunto alla lista di dati corrotti

with open('dataframe_0.pkl', 'wb') as handle:
    pkl.dump(dataframe, handle, protocol=pkl.HIGHEST_PROTOCOL)


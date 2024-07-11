import pickle as pkl
import pandas as pd


dataframe = pkl.load(open('quadranti/quadrante_1/plc_quadrante_1.pkl', 'rb'))
excel_file_path = 'best_params_plc_quadrante_1.xlsx'


# Inizializzazione della lista per raccogliere i dati dei parametri
param_data = []
i = 1
for user in dataframe:
    print(f"Best params per Utente {i}")
    i = i + 1
    best_params = dataframe[user][3].best_params_

    # Creazione di un dizionario per contenere i dati dei parametri per l'utente corrente
    user_param_data = {'Utente': user}

    # Aggiunta delle informazioni sui parametri (nome del parametro e valore)
    for param, value in best_params.items():
        user_param_data[param] = value
        print(f"{param}: {value}")
    print("\n")

    # Aggiunta del dizionario alla lista
    param_data.append(user_param_data)

# Creazione di un DataFrame pandas dai dati raccolti
param_df = pd.DataFrame(param_data)

# Stampa del DataFrame
print(param_df)

# Esporta il DataFrame su Excel
param_df.to_excel(excel_file_path, index=False)

print(f"DataFrame esportato su Excel: {excel_file_path}")
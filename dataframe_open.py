import pickle as pkl

dataframe = pkl.load(open('features/utente/utente_stimuli.pkl', 'rb'))
print(dataframe)
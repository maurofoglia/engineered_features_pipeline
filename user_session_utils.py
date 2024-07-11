def correct_session(data):
    if data[0]['time_series'].shape[1] == 5:
        marker = data[1]
        eeg = data[0]

        data[0] = marker
        data[1] = eeg

    return data
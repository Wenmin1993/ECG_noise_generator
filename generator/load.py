import wfdb
import numpy as np
import matplotlib.pyplot as plt

signal, fields=wfdb.rdsamp('ma',channels=[0, 1], sampfrom=0, sampto=1500,pn_dir='nstdb')
#mitdb数据库的是数据是两导联，格式是[650000,2]的数据，channels是选择读取哪一个导联的
print('signal:',signal)
print('fields:',fields['fs'])

record = wfdb.rdrecord('bw',pn_dir='nstdb/')
annotation = wfdb.rdann('bw','atr',sampto=3600,pn_dir='nstdb/')
wfdb.plot_wfdb(record=record,annotation=annotation,title='Record bw from MIT-BIH Noise Stress Test Database')

plt.plot(signal)
plt.ylabel(fields['units'][0])
plt.legend([fields['sig_name'][0],fields['sig_name'][1]])
plt.show()




def get_noise_records(database,noise_type):
    record = wfdb.rdrecord(noise_type,sampfrom=0,sampto=3600,pn_dir=database)
    #record= wfdb.rdrecord(noise_type,sampto=3600,pn_dir=database)
    annotation = wfdb.rdann(noise_type,'atr',sampto=3600,pn_dir=database)

    wfdb.plot_wfdb(record=record,annotation=annotation,title='Record'+noise_type+' from MIT-BIH Noise Stress Test Database')
    print('annotation:', annotation.__dict__)
    print('record:', record.__dict__)
    print('signal:', record.p_signal)  # 这个record其实并不是字典需要用点操作符取出值
    return record









def get_records(db_folder):
    #db_folder ='mitdb'
    record_files =wfdb.get_record_list(db_folder)
    print(record_files[0])
    data = wfdb.rdsamp('{}/{}'.format(db_folder,record_files[0]))
    records = [wfdb.rdsamp('{}/{}'.format(db_folder,record))for record in record_files]

    print("number of records: ", len(record_files))
    print("record_files:", record_files)
    print("records :",records)
    print(data)
    print(np.shape(data[0]))
    return records

def data_from_records(records, channel, db):
    """
    Extract ECG, beat locations and beat types from Physionet database.

    Takes a list of record names, ECG channel index and name of the
    PhysioNet data base. Tested only with db == 'mitdb'.

    Parameters
    ----------
    records : list
        list of file paths to the wfdbdb-records
    channel : int
        ECG channel that is wanted from each record
    db : string
        Name of the PhysioNet ECG database

    Returns
    -------
    signals : list
        list of single channel ECG records stored as numpy arrays
    beat_locations : list
        list of numpy arrays where each array stores beat locations as
        samples from the beg of one resampled single channel
        ECG recording
    beat_types : list
        list of numpy arrays where each array stores the information of
        the beat types for the corresponding array in beat_locations

    """
    signals = []
    beat_locations = []
    beat_types = []

    for record in records:
        print('processing record: ', record)
        signal = (rdsamp(record, pb_dir=db))
        signal_fs = signal[1]['fs']
        annotation = rdann(record, 'atr', pb_dir=db)

        # resample to 250 Hz
        signal, annotation = resample_singlechan(
                                signal[0][:, channel],
                                annotation,
                                fs=signal_fs,
                                fs_target=250)

        beat_loc, beat_type = get_beats(annotation)

        signals.append(signal)
        beat_locations.append(beat_loc)
        beat_types.append(beat_type)

    return signals, beat_locations, beat_types

def create_sine(sampling_frequency, time_s, sine_frequency):
    """
    Create sine wave.

    Function creates sine wave of wanted frequency and duration on a
    given sampling frequency.

    Parameters
    ----------
    sampling_frequency : float
        Sampling frequency used to sample the sine wave
    time_s : float
        Lenght of sine wave in seconds
    sine_frequency : float
        Frequency of sine wave

    Returns
    -------
    sine : array
        Sine wave

    """
    samples = np.arange(time_s * sampling_frequency) / sampling_frequency
    sine = np.sin(2 * np.pi * sine_frequency * samples)
    print(sine)
    return sine

def ecg_generator(signals, peaks, labels, ma, bw, win_size, batch_size):
    """
        Generate ECG data with R-peak labels.

        Data generator that yields training data as batches. Every instance
        of training batch is composed as follows:
        1. Randomly select one ECG signal from given list of ECG signals
        2. Randomly select one window of given win_size from selected signal
        3. Check that window has at least one beat and that all beats are
           labled as normal
        4. Create label window corresponding the selected window
            -beats and four samples next to beats are labeled as 1 while
             rest of the samples are labeled as 0
        5. Normalize selected signal window from -1 to 1
        6. Add noise into signal window and normalize it again to (-1, 1)
        7. Add noisy signal and its labels to trainig batch
        8. Transform training batches to arrays of needed shape and yield
           training batch with corresponding labels when needed

        Parameters
        ----------
        signals : list
            List of ECG signals
        peaks : list
            List of peaks locations for the ECG signals
        labels : list
            List of labels (peak types) for the peaks
        ma : array
            Muscle artifact signal
        bw : array
            Baseline wander signal
        win_size : int
            Number of time steps in the training window
        batch_size : int
            Number of training examples in the batch

        Yields
        ------
        (X, y) : tuple
            Contains training samples with corresponding labels

        """
    while True:

            X = []
            y = []

            while len(X) < batch_size:
                random_sig_idx = np.random.randint(0, len(signals))
                random_sig = signals[random_sig_idx]
                p4sig = peaks[random_sig_idx]
                plabels = labels[random_sig_idx]

                # Select one window
                beg = np.random.randint(random_sig.shape[0] - win_size)
                end = beg + win_size

                # Select peaks that fall into selected window.
                # Buffer of 3 to the window edge is needed as labels are
                # inserted also next to point)
                p_in_win = p4sig[(p4sig >= beg + 3) & (p4sig <= end - 3)] - beg

                # Check that there is at least one peak in the window
                if p_in_win.shape[0] >= 1:

                    # Select labels that fall into selected window
                    lab_in_win = plabels[(p4sig >= beg + 3) & (p4sig <= end - 3)]

                    # Check that every beat in the window is normal beat
                    if np.all(lab_in_win == 1):
                        # Create labels for data window
                        window_labels = np.zeros(win_size)
                        np.put(window_labels, p_in_win, lab_in_win)

                        # Put labels also next to peak
                        np.put(window_labels, p_in_win + 1, lab_in_win)
                        np.put(window_labels, p_in_win + 2, lab_in_win)
                        np.put(window_labels, p_in_win - 1, lab_in_win)
                        np.put(window_labels, p_in_win - 2, lab_in_win)

                        # Select data for window and normalize it (-1, 1)
                        data_win = normalize_bound(random_sig[beg:end],
                                                   lb=-1, ub=1)

                        # Add noise into data window and normalize it again
                        data_win = data_win + get_noise(ma, bw, win_size)
                        data_win = normalize_bound(data_win, lb=-1, ub=1)

                        X.append(data_win)
                        y.append(window_labels)

            X = np.asarray(X)
            y = np.asarray(y)

            X = X.reshape(X.shape[0], X.shape[1], 1)
            y = y.reshape(y.shape[0], y.shape[1], 1).astype(int)

            yield (X, y)


#ma = get_noise_records('nstdb/','ma')
#bw = get_noise_records('nstdb/','bw')

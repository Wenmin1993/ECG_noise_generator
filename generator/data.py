from random import uniform
from typing import List, Any
import matplotlib.pyplot as plt
import wfdb
from wfdb.processing import resample_singlechan, find_local_peaks, correct_peaks, normalize_bound
import numpy as np

def get_noise_record(noise_type, database):
    print('processing noise:',noise_type)
    signal,field =wfdb.rdsamp(noise_type,channels=[0],pn_dir=database)
    signal_size = field['sig_len']
    print('-----noise processed!-----')
    plt.plot(signal)
    plt.ylabel(field['units'][0])
    plt.legend([field['sig_name'][0]])
    plt.title('Sample of '+noise_type+' from '+database )
    plt.show()
    return signal,field,signal_size

def fix_labels(signals, beats, labels):
    """
    Change labeling of the normal beats.

    Beat index of some normal beats doesn't occur at the local maxima
    of the ECG signal in MIT-BIH Arrhytmia database. Function checks if
    beat index occurs within 5 samples from the local maxima. If this is
    not true, beat labeling is changed to -1.

    Parameters
    ----------
    signals : list
        List of ECG signals as numpy arrays
    beats : list
        List of numpy arrays that store beat locations
    labels : list
        List of numpy arrays that store beat types

    Returns
    -------
    fixed_labels : list
        List of numpy arrays where -1 has been added for beats that are
        not located in local maxima

    """
    fixed_labels = []
    for s, b, l in zip(signals, beats, labels):

        # Find local maximas
        localmax = find_local_peaks(sig=s, radius=5)
        localmax = correct_peaks(sig=s,
                                 peak_inds=localmax,
                                 search_radius=5,
                                 smooth_window_size=20,
                                 peak_dir='up')

        # Make sure that beat is also in local maxima
        fixed_p = correct_peaks(sig=s,
                                peak_inds=b,
                                search_radius=5,
                                smooth_window_size=20,
                                peak_dir='up')

        # Check what beats are in local maximas
        beat_is_local_peak = np.isin(fixed_p, localmax)
        fixed_l = l

        # Add -1 if beat is not in local max
        fixed_l[~beat_is_local_peak] = -1
        fixed_labels.append(fixed_l)

    return fixed_labels

def get_white_Gaussian_Noise(signal,snr):
    snr = 10 ** (snr / 10.0)
    print(len(signal))
    power_signal = np.sum(signal ** 2) / len(signal)
    power_noise = power_signal / snr
    wgn = np.random.randn(len(signal)) * np.sqrt(power_noise)

    plt.subplot(211)
    plt.title('Gauss Distribution')
    plt.hist(wgn, bins=100)
    plt.subplot(212)
    plt.plot(wgn)
    plt.show()

    return wgn

def get_noise(wgn,ma,bw,win_size):

    # Get the slice of data
    print('shape:',ma.shape[0],bw.shape[0],wgn.shape[0])
    beg = np.random.randint(ma.shape[0]-win_size)
    end = beg+win_size
    beg2 = np.random.randint(bw.shape[0]-win_size)
    end2 = beg2+win_size
    beg3 = np.random.randint(wgn.shape[0]-win_size)
    end3 = beg3+win_size

    mains = creat_sine(128,int(win_size/128),60)*uniform(0,0.5)

    mode = np.random.randint(6)
    ma_multip = uniform(0,5)
    bw_multip = uniform(0,10)
    print(mode)

    # Add noise
    if mode == 0:
        noise = ma[beg:end]*ma_multip
    elif mode == 1:
        noise = bw[beg2:end2]*bw_multip
    elif mode == 2:
        noise = wgn[beg3:end3]
    elif mode == 3:
        noise = ma[beg:end]*ma_multip+wgn[beg3:end3]
    elif mode == 4:
        noise = bw[beg2:end2]*bw_multip+wgn[beg3:end3]
    elif mode == 5:
        noise = ma[beg:end]*ma_multip+bw[beg2:end2]*bw_multip
    else:
        noise = ma[beg:end]*ma_multip+bw[beg2:end2]*bw_multip+wgn[beg3:end3]


    noise_final = noise + mains
    print('noise:',noise_final)
    plt.subplot(211)
    plt.plot(mains)
    plt.title('Sine wave')
    plt.subplot(212)
    plt.plot(noise_final)
    plt.title('final noise')
    plt.show()
    return noise_final

def get_beats(annotation):
    """
       Extract beat indices and types of the beats.

       Beat indices indicate location of the beat as samples from the
       beg of the signal. Beat types are standard character
       annotations used by the PhysioNet.

       Parameters
       ----------
       annotation : wfdbdb.io.annotation.Annotation
           wfdbdb annotation object

       Returns
       -------
       beats : array
           beat locations (samples from the beg of signal)
       symbols : array
           beat symbols (types of beats)

     """
    # All beat annotations
    beat_annotations = ['N', 'L', 'R', 'B', 'A',
                        'a', 'e', 'J', 'V', 'r',
                        'F', 'S', 'j', 'n', 'E',
                        '/', 'Q', 'f', '?']

    # Get indices and symbols of the beat annotations
    indices = np.isin(annotation.symbol, beat_annotations)
    symbols = np.asarray(annotation.symbol)[indices]
    beats = annotation.sample[indices]
    #print(annotation.sample)
    return beats, symbols

def creat_sine(sampling_frequency,time_s,sine_frequency):
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
    print('samples:',samples)
    sine = np.sin(2 * np.pi * sine_frequency * samples)
    print('sine',sine)
    plt.plot(sine)
    plt.title('sine')
    plt.show()
    return sine

def get_ecg_records(database, channel):
    signals = []
    beat_locations = []
    beat_types = []
    useless_afrecord = ['00735','03665','04043','08405','08434',]

    record_files = wfdb.get_record_list(database)
    print('record_files:',record_files)

    for record in record_files:
        if record in useless_afrecord:
            continue
        else:
            print('processing record:',record)
            s, f = wfdb.rdsamp(record, pn_dir=database)
            print(f)

            #print('length of signal: ',f['sig_len'])
            #sample_from = np.random.randint(f['sig_len']-1000)
            #print('sample_from: ',sample_from)
            #signal = (wfdb.rdsamp(record,channels=[channel], sampfrom=sample_from, sampto=sample_from+1000, pn_dir=database))
            annotation = wfdb.rdann('{}/{}'.format(database, record), extension='atr')
            if f['fs'] != 128:
                signal, annotation = resample_singlechan(s[:,channel],annotation, fs=f['fs'], fs_target=128)
            else:
                signal, field = wfdb.rdsamp(record,channels= [channel], pn_dir=database)

            print(signal)
            beat_loc, beat_type = get_beats(annotation)
            signals.append(signal)
            beat_locations.append(beat_loc)
            beat_types.append(beat_type)
            print('size of signal list: ',len(signals))
            print('--------')

    print('first signal in list :',signals[0])
    print('-------record processed!------')
    plt.plot(signals[0])
    plt.title(database+' record')
    plt.show()


    return signals, beat_locations, beat_types

    """
    records =[wfdb.rdsamp('{}/{}'.format(database,record)) for record in record_files]
    annotations = [wfdb.rdann('{}/{}'.format(database,record),extension='atr') for record in record_files]
    signal, annotation = resample_singlechan(signal[0][:, channel],annotation,fs=signal_fs,fs_target=250)
    data =wfdb.rdsamp('{}/{}'.format(database,record_files[0]))


    print("number of records: ", len(record_files))
    print(np.shape(data[0]))
    print(data)
    print('records:',records)

    print('annotation:',annotations)   
    """

def ecg_generator(signals,peaks,labels,wgn,ma,bw,win_size,batch_size):
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
                    data_win = data_win + get_noise(wgn,ma, bw, win_size)
                    data_win = normalize_bound(data_win, lb=-1, ub=1)

                    X.append(data_win)
                    y.append(window_labels)

        X = np.asarray(X)
        y = np.asarray(y)

        X = X.reshape(X.shape[0], X.shape[1], 1)
        y = y.reshape(y.shape[0], y.shape[1], 1).astype(int)

        yield (X, y)

nsr,nsr_bls,nsr_labels = get_ecg_records('nsrdb', 0)
af,af_bls,af_labels = get_ecg_records('afdb', 0)
ma,ma_field,ma_size = get_noise_record('ma','nstdb')
bw,bw_field,bw_size= get_noise_record('bw','nstdb')

'''
#test for get_white_Gaussian_Noise(x).

t = np.arange(0, 1000000) * 0.1
x = np.sin(t)
wgn = get_white_Gaussian_Noise(x,6)
xn = x+wgn
plt.subplot(311)
plt.title('Gauss Distribution')
plt.hist(wgn, bins=100)
plt.subplot(312)
plt.psd(wgn)
plt.subplot(313)
plt.psd(xn)
plt.show()
'''
wgn_nsr = get_white_Gaussian_Noise(nsr[0],6)
wgn_af = get_white_Gaussian_Noise(af[0],6)

noise_for_nsr = get_noise(wgn_nsr,ma,bw,1280)
noise_for_af = get_noise(wgn_af,ma,bw,1280)

ecg_generator(nsr,)

'''
sine = creat_sine(128,int(1280/128),60)
print(sine.shape[0])
'''


# print(record.__dict__)

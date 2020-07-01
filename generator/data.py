from random import uniform
from typing import List, Any
import matplotlib.pyplot as plt
import wfdb
from wfdb.processing import resample_sig,resample_singlechan, find_local_peaks, correct_peaks, normalize_bound
import numpy as np
import pickle


def serialization(name,content):
    f = open(r'pickle\s_' +name,'wb')
    pickle.dump(content,f)
    f.close()
def deserialization(name):
    f = open(r'pickle\s_'+name, 'rb')
    x = pickle.load(f)
    f.close()
    return x
def get_ecg_records(database, channel):

    signals = []
    beat_locations = []
    beat_types = []
    useless_afrecord = ['00735','03665','04043','08405','08434']

    record_files = wfdb.get_record_list(database)
    print('record_files:',record_files)

    for record in record_files:
        if record in useless_afrecord:
            continue
        else:
            print('processing record:',record)
            s, f = wfdb.rdsamp(record, pn_dir = database)
            print(f)
            annotation = wfdb.rdann('{}/{}'.format(database, record), extension='atr')
            signal, annotation = resample_singlechan(s[:,channel],annotation, fs=f['fs'], fs_target=128)

            print(signal)
            print(signal.shape)
            beat_loc, beat_type = get_beats(annotation)
            signals.append(signal)
            beat_locations.append(beat_loc)
            beat_types.append(beat_type)
            print('size of signal list: ',len(signals))
            print('--------')
    print('---------record processed!---')
    serialization(database + '_signal', signals)
    serialization(database + '_beat_loc', beat_locations)
    serialization(database + '_beat_types', beat_types)

    return signals, beat_locations, beat_types
def load_ecg_records(database):
    print('-----loading----- ')
    signals = deserialization(database + '_signal')
    beat_locations = deserialization(database + '_beat_loc')
    beat_types = deserialization(database + '_beat_types')
    print('singnals.shape:',np.asarray(signals).shape)
    print('-------ecg record from '+database+'loaded!------')
    return signals,beat_locations,beat_types
def get_noise_record(noise_type, database):

    print('processing noise:',noise_type)
    s,f= wfdb.rdsamp(noise_type,channels=[0],pn_dir=database)

    print(s)
    print(f)
    signal,_ = resample_sig(s[:,0],fs=360,fs_target=128)
    print(signal)
    signal_size = f['sig_len']
    print('-----noise processed!-----')

    #serialization the data
    serialization(database + '_' + noise_type + '_signal', signal)
    serialization(database + '_' + noise_type+ '_field', f)
    serialization(database + '_' + noise_type + '_size', signal_size)

    return signal,f,signal_size
def load_noise_signal(database,noise_type):
    # deserialization the data
    signal = deserialization(database + '_' + noise_type + '_signal')
    field = deserialization(database + '_' + noise_type + '_field')
    signal_size = deserialization(database + '_' + noise_type + '_size')
    print(noise_type+' singnals.shape:',np.asarray(signal).shape)
    print('-------'+noise_type+' noise signal loaded!------')

    return signal,field,signal_size
def get_white_Gaussian_Noise(noise_type,signal,snr):
    snr = 10 ** (snr / 10.0)
    print(len(signal))
    power_signal = np.sum(signal ** 2) / len(signal)
    power_noise = power_signal / snr
    wgn = np.random.randn(len(signal)) * np.sqrt(power_noise)
    serialization(noise_type+'_wgn_noise',wgn)
    return wgn
def load_wgn_noise(noise_type):
    wgn = deserialization(noise_type+'_wgn_noise')
    print(noise_type+'_wgn.shape:',wgn.shape)
    return wgn
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
    sine = np.sin(2*sine_frequency*np.pi*np.arange(time_s*sampling_frequency)/sampling_frequency)
    '''
    n=np.arange(0,time_s,1/sampling_frequency)
    plt.plot(n,sine)
    plt.title('sine')
    plt.show()
    '''
    return sine
def get_noise(name,wgn,ma,bw,win_size):

    # Get the slice of data
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
    print('noise mode for '+name+ ':',mode)

    '''
    #test for noise
    noise_0 = ma[beg:end]*ma_multip
    noise_1 = bw[beg2:end2] * bw_multip
    noise_2 = wgn[beg3:end3]
    noise_3 = ma[beg:end] * ma_multip + wgn[beg3:end3]
    noise_4 = bw[beg2:end2] * bw_multip + wgn[beg3:end3]
    noise_5 = ma[beg:end] * ma_multip + bw[beg2:end2] * bw_multip
    noise_6 = ma[beg:end] * ma_multip + bw[beg2:end2] * bw_multip + wgn[beg3:end3]

    
    plt.subplot(711)
    plt.plot(noise_0)
    plt.title('noise_0')
    plt.subplot(712)
    plt.plot(noise_1)
    plt.title('noise_1')
    plt.subplot(713)
    plt.plot(noise_2)
    plt.title('noise_2')
    plt.subplot(714)
    plt.plot(noise_3)
    plt.title('noise_3')
    plt.subplot(715)
    plt.plot(noise_4)
    plt.title('noise_4')
    plt.subplot(716)
    plt.plot(noise_5)
    plt.title('noise_5')
    plt.subplot(717)
    plt.plot(noise_6)
    plt.title('noise_6')
    plt.show()
    '''

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
    #else:
    elif mode == 5:
        noise = ma[beg:end]*ma_multip+bw[beg2:end2]*bw_multip
    else:
        noise = ma[beg:end]*ma_multip+bw[beg2:end2]*bw_multip+wgn[beg3:end3]
    
    noise_final = noise + mains
    #print('shape: ma:',ma.shape,',bw:',bw.shape,',wgn:',wgn.shape,',mains:',mains.shape,',fianl:',noise_final.shape)
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
def ecg_generator(name,signals,wgn,ma,bw,win_size,batch_size):

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
    print('processing')
    while True:
        x = []
        section = []
        noise_list = []
        while len(x) < batch_size:
            random_sig_idx = np.random.randint(0, len(signals))

            random_sig = signals[random_sig_idx]

            # Select one window
            beg = np.random.randint(random_sig.shape[0] - win_size)
            end = beg + win_size
            section.append(random_sig[beg:end])

            # Select data for window and normalize it (-1, 1)
            data_win = normalize_bound(random_sig[beg:end],lb=-1, ub=1)

            # Add noise into data window and normalize it again
            added_noise =  get_noise(name,wgn,ma,bw,win_size)
            noise_list.append(added_noise)
            data_win = data_win + added_noise
            data_win = normalize_bound(data_win, lb=-1, ub=1)
            x.append(data_win)

        x = np.asarray(x)
        x = x.reshape(x.shape[0], x.shape[1], 1)
        id = np.random.randint(0,len(x))
        plt.subplot(311)
        plt.plot(section[id])
        plt.title('original '+name+' ecg')
        plt.subplot(312)
        plt.plot(x[id])
        plt.title('noised '+name+' ecg')
        plt.subplot(313)
        plt.title('added noise')
        plt.plot(noise_list[id])
        #plt.savefig(name+'.png')
        plt.show()

        print(x.shape)
        serialization(name+'_noised_ecg',x)
        return x

'''
#serialization of nsr and af ecg records 
nsr,nsr_bls,nsr_labels = get_ecg_records('nsrdb', 0)
af,af_bls,af_labels = get_ecg_records('afdb', 0)

#serialization of ma and bw noise
ma,ma_field,ma_size = get_noise_record('ma','nstdb')
bw,bw_field,bw_size= get_noise_record('bw','nstdb')
'''
#deserialization of ecg records
nsr,nsr_bls,nsr_labels =load_ecg_records('nsrdb')
af,af_bls,af_labels = load_ecg_records('afdb')

#deserialization of noise
ma,ma_field,ma_size = load_noise_signal('nstdb','ma')
bw,bw_field,bw_size = load_noise_signal('nstdb','bw')

'''
#serialization of noise signal
id_nsr = np.random.randint(len(nsr))
id_af  = np.random.randint(len(af))
wgn_nsr = get_white_Gaussian_Noise('nsr',nsr[id_nsr],6)
wgn_af = get_white_Gaussian_Noise('af',af[id_af],6)
'''

#deserialization of noise_signal
wgn_nsr = load_wgn_noise('nsr')
wgn_af = load_wgn_noise('af')

#generate noised ecg signals and serialization
noised_nsr = ecg_generator('nsr',nsr,wgn_nsr,ma,bw,win_size=1280,batch_size=256)
noised_af  = ecg_generator('af',af,wgn_af,ma,bw,win_size=1280,batch_size=256)

#deserialization noised ecg signals
noised_nsr = deserialization('nsr_noised_ecg')
noised_af  = deserialization('af_noised_ecg')





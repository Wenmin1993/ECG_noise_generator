from random import uniform
from typing import List, Any
import matplotlib.pyplot as plt
import wfdb
from wfdb.processing import resample_singlechan, find_local_peaks, correct_peaks, normalize_bound
import numpy as np

def get_noise_record(noise_type, database):
    signal,field =wfdb.rdsamp(noise_type,channels=[0],pn_dir=database)
    signal_size = field['sig_len']
    plt.plot(signal)
    plt.ylabel(field['units'][0])
    plt.legend([field['sig_name'][0]])
    plt.title('Sample of '+noise_type+' from '+database )
    plt.show()
    return signal,field,signal_size

def add_GaussianNoise():

    return noise

def get_noise(ma,bw,win_size):
    # Get the slice of data
    #print(ma.shape[0],bw.shape[0])
    beg = np.random.randint(ma.shape[0]-win_size)
    end = beg+win_size
    beg2 = np.random.randint(bw.shape[0]-win_size)
    end2 = beg2+win_size

    mains = creat_sine(128,int(win_size/128),60)*uniform(0,0.5)
    mode = np.random.randint(3)
    ma_multip = uniform(0,5)
    bw_multip = uniform(0,10)

    # Add noise
    if mode == 0:
        noise = ma[beg:end]*ma_multip
    elif mode == 1:
        noise = bw[beg:end]*bw_multip
    else:
        noise = (ma[beg:end]*ma_multip)+(bw[beg2:end2]*bw_multip)


    noise_final = noise + mains
    print('noise',noise_final)
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
    sine = np.sin(2 * np.pi * sine_frequency * samples)
    print('sine',sine)
    plt.plot(sine)
    plt.show()
    return sine

def get_ecg_records(database, channel):
    signals = []
    beat_locations = []
    beat_types = []

    record_files = wfdb.get_record_list(database)
    print(record_files)

    for record in record_files:
        print('processing record:',record)
        s, f = wfdb.rdsamp(record, channels=[channel], pn_dir=database)
        sample_from = np.random.randint(f['sig_len']-1000)
        signal = ( wfdb.rdsamp(record,channels=[channel], sampfrom=sample_from, sampto=sample_from+1000, pn_dir=database))
        annotation = wfdb.rdann('{}/{}'.format(database, record), extension='atr')
        #signal, annotation = resample_singlechan(sig[:, channel], annotation, fs=signal_fs, fs_target=120)

        beat_loc, beat_type = get_beats(annotation)
        signals.append(signal)
        beat_locations.append(beat_loc)
        beat_types.append(beat_type)
        print(len(signals))

    print(signals[0][0])
    plt.plot(signals[0][0])
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

def ecg_generator(signals,peaks,)

ma,ma_field,ma_size = get_noise_record('ma','nstdb')
bw,bw_field,bw_size= get_noise_record('bw','nstdb')
#print('ma:',ma,'ma_size:',ma_size,'bw:',bw,'bw_size:',bw_size)
noise = get_noise(ma,bw,1000)
nsr = get_ecg_records('nsrdb', 0)
#sine_wave = creat_sine(128,1000,60)


# record = wfdb.rdrecord('bw',pn_dir='nstdb/')
# wfdb.plot_wfdb(record=record, title='Record bw in nstdb') #显示该记录
# print(record.__dict__)#打印该记录的完成字典信息

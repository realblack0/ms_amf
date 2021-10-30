from scipy.io import loadmat
from scipy.signal import butter,lfilter
import pandas as pd
import numpy as np


def load_gdf2mat(subject, train=True, data_dir=".", overflowdetection=False):
    # Configuration
    if train:
        filename = f"A0{subject}T_gdf"
    else:
        filename = f"A0{subject}E_gdf"
    base = data_dir

    # Load mat files
    print("\nLoad data")
    data_path  =  base + '/gdf2mat/' + filename + '.mat'
    label_path =  base + '/true_labels/' + filename[:4] + '.mat'

    if overflowdetection:
        filename  = filename + "_overflowdetection_off"
        data_path = base + '/gdf2mat_overflowdetection_off/' + filename + '.mat'
    
    data       = loadmat(data_path, squeeze_me=False)
    label_data = loadmat(label_path, squeeze_me=False) 

    # Parse data
    s = data["s"] # signal
    h = data["h"] # header
    labels = label_data["classlabel"] # true label

    h_names = h[0][0].dtype.names # header is structured array
    origin_filename = h["FileName"][0,0][0]
    train_labels = h["Classlabel"][0][0] # For Evaluation data, it is filled with NaN.
    artifacts = h["ArtifactSelection"][0,0]

    events = h['EVENT'][0,0][0,0] # void
    typ = events['TYP']
    pos = events['POS']
    fs  = events['SampleRate'].squeeze()
    dur = events['DUR']

    # http://www.bbci.de/competition/iv/desc_2a.pdf
    typ2desc = {276:'Idling EEG (eyes open)',
                277:'Idling EEG (eyes closed)',
                768:'Start of a trial',
                769:'Cue onset left (class 1)',
                770:'Cue onset right (class 2)',
                771:'Cue onset foot (class 3)',
                772:'Cue onset tongue (class 4)',
                783:'Cue unknown',
                1024:'Eye movements',
                32766:'Start of a new run'}

    # 출처... 아마... brain decode...
    ch_names = ['Fz',  'FC3', 'FC1', 'FCz', 'FC2',
                 'FC4', 'C5',  'C3',  'C1',  'Cz',
                 'C2',  'C4',  'C6',  'CP3', 'CP1',
                 'CPz', 'CP2', 'CP4', 'P1',  'Pz',
                 'P2',  'POz', 'EOG-left', 'EOG-central', 'EOG-right']

    print("- filename:", filename)
    print("- load data from:", data_path)
    print('\t- original fileanme:', origin_filename)
    print("- load label from:", label_path)
    assert filename[:4] == origin_filename[:4]

    print("- shape of s", s.shape) # (time, 25 channels), 
    print("- shape of labels", labels.shape) # (288 trials)
    
    return {"s":s, "h":h, "labels":labels, "filename":filename, "artifacts":artifacts, "typ":typ, "pos":pos, "fs":fs, "dur":dur, "typ2desc":typ2desc, "ch_names":ch_names}
      
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order, axis=-1):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = lfilter(b, a, data, axis=axis)
    return y
    
def preprocess(s, labels, ch_names, pos, typ, fs, artifacts, filename, reject_trial="auto", **kwargs):
    """
    1. Fill NaN using linear interpolation
    2. Bandpass 8-30Hz
    3. Split trials
    4. Remove rejected trials with artifact
    """
    # FILL NAN VALUES
    print("\nLinear interpolation")
    s_df = pd.DataFrame(data=s, columns=ch_names) # there are a lots of NaN values
    print(" - # NaN values (before interpolation)")
    print(s_df.isna().sum())
    s_df = s_df.interpolate(method='linear', axis=0) # linear interpolation
    print("\n - # NaN values (after interpolation)")
    print(s_df.isna().sum())
    
    # BANDPASS FILTER
    print("\nBandpass 8-30Hz")
    s_bp = butter_bandpass_filter(s_df.values.T, lowcut=8, highcut=30, fs=fs, order=4, axis=1)
    print("- shape of s_bp", s_bp.shape)
    
    # SPLIT TRIALS
    print("\nEpoching")
    start_onset = pos[typ == 768] # start of a trial
    trials = []
    for i, onset in enumerate(start_onset):
        trials.append(s_bp[0:22, onset+3*fs:onset+5*fs]) # time : start+3 ~ start+5 sec, channel : 22 EEG channels
    trials = np.array(trials) # trials, channels, time
    print("- shape of trials", trials.shape)
    print("- shape of labels", labels.shape)
    
    # REMOVE REJECTED TRIALS
    print("\nRemove rejected trials with artifact")
    if reject_trial=="auto":
        if filename[3] == "T":
            artifact_mask = ~ artifacts.squeeze().astype(bool) # False : rejected trial 
            trials = trials[artifact_mask]
            labels = labels[artifact_mask]
        elif filename[3] == "E":
            print("- Skip for Evaluation data.")
    elif reject_trial:
        artifact_mask = ~ artifacts.squeeze().astype(bool) # False : rejected trial 
        trials = trials[artifact_mask]
        labels = labels[artifact_mask]
    else:
        print("- Skip")
    print("- shape of trials", trials.shape)
    print("- shape of labels", labels.shape)
    
    return trials, labels
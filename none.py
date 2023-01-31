#!/usr/bin/env python
# coding: utf-8

# 
# 1. Data load
#     1. load
#     2. s to cnt
#     3. label index (1\~4 -> 0\~3)
#     4. pos index (start from 0)
# 2. Preprocessing
#     1. drop EOG channels
#     2. replace break with mean
#     3. scaling (microvolt)
#     4. bandpass 4-38Hz (butterworth 3rd order)
#     5. exponential running standardization (init_block_size=1000, factor_new=1e-3)
#     6. epoch (cue-0.5ms ~ cue+4ms)
#     * no rejection
# 3. Split
#     1. split train into train and validation (8:2)
# 4. Crop (bunch of crops)
#     1. input_time_length: 1000 samples
#     * augmentation effect (twice)
# 5. Data loader
# 6. Model
#     - ShallowNet
#     - to_dense_prediction_model
#     - xavier initialization
# 7. Learning strategy
#     - loss function 
#         1. log softmax + NLLloss for each crop (=CrossEntropyLoss)
#         2. tied sample loss
#     - optimizer : Adam
#     - evaluater for cropped learning
#     - early stop 
#         1. using training, no decrease on val acc (80 epoch) or max epoch 800
#         2. using training and val, same training loss with val loss from first stop or max epoch 800.
#     - maxnorm

# 1. Data load
#     1. load
#     2. s to cnt
#     3. label index (1\~4 -> 0\~3)
#     4. pos index (start from 0)

# In[1]:


from scipy.io import loadmat
import mne
import numpy as np
from copy import deepcopy
from functools import wraps

def verbose_func_name(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        if ("verbose" in kwargs.keys()) and kwargs["verbose"]:
            print("\n"+fn.__name__)
        return fn(*args, **kwargs)
    return inner

# @verbose_func_name
# def load_gdf2mat(subject, train=True, data_dir=".", overflowdetection=True, verbose=False):
#     # Configuration
#     if train:
#         filename = f"A0{subject}T_gdf"
#     else:
#         filename = f"A0{subject}E_gdf"
#     base = data_dir

#     # Load mat files
#     data_path  =  base + '/gdf2mat/' + filename + '.mat'
#     label_path =  base + '/true_labels/' + filename[:4] + '.mat'

#     if not overflowdetection:
#         filename  = filename + "_overflowdetection_off"
#         data_path = base + '/gdf2mat_overflowdetection_off/' + filename + '.mat'
    
#     session_data = loadmat(data_path, squeeze_me=False)
#     label_data   = loadmat(label_path, squeeze_me=False) 

#     # Parse data
#     s = session_data["s"] # signal
#     h = session_data["h"] # header
#     labels = label_data["classlabel"] # true label

#     h_names = h[0][0].dtype.names # header is structured array
#     origin_filename = h["FileName"][0,0][0]
#     train_labels = h["Classlabel"][0][0] # For Evaluation data, it is filled with NaN.
#     artifacts = h["ArtifactSelection"][0,0]

#     events = h['EVENT'][0,0][0,0] # void
#     typ = events['TYP']
#     pos = events['POS']
#     fs  = events['SampleRate'].squeeze()
#     dur = events['DUR']

#     # http://www.bbci.de/competition/iv/desc_2a.pdf
#     typ2desc = {276:'Idling EEG (eyes open)',
#                 277:'Idling EEG (eyes closed)',
#                 768:'Start of a trial',
#                 769:'Cue onset left (class 1)',
#                 770:'Cue onset right (class 2)',
#                 771:'Cue onset foot (class 3)',
#                 772:'Cue onset tongue (class 4)',
#                 783:'Cue unknown',
#                 1024:'Eye movements',
#                 32766:'Start of a new run'}

#     # 출처... 아마... brain decode...
#     ch_names = ['Fz',  'FC3', 'FC1', 'FCz', 'FC2',
#                  'FC4', 'C5',  'C3',  'C1',  'Cz',
#                  'C2',  'C4',  'C6',  'CP3', 'CP1',
#                  'CPz', 'CP2', 'CP4', 'P1',  'Pz',
#                  'P2',  'POz', 'EOG-left', 'EOG-central', 'EOG-right']

#     assert filename[:4] == origin_filename[:4]
#     if verbose:
#         print("- filename:", filename)
#         print("- load data from:", data_path)
#         print('\t- original fileanme:', origin_filename)
#         print("- load label from:", label_path)
#         print("- shape of s", s.shape) # (time, 25 channels), 
#         print("- shape of labels", labels.shape) # (288 trials)

#     data =  {"s":s, "h":h, "labels":labels, "filename":filename, "artifacts":artifacts, "typ":typ, "pos":pos, "fs":fs, "dur":dur, "typ2desc":typ2desc, "ch_names":ch_names}
#     return data

@verbose_func_name
def load_gdf2mat_feat_mne(subject, train=True, data_dir=".", 
                          overflowdetection=True, verbose=False):
    # Configuration
    if train:
        filename = f"A0{subject}T_gdf"
    else:
        filename = f"A0{subject}E_gdf"
    base = data_dir

    assert not overflowdetection, (
        "load_gdf2mat_feat_mne does not support overflowdetection...")
    
    # Load mat files
    data_path  =  base + '/gdf2mat_overflowdetection_off/'                   + filename + '_overflowdetection_off.mat'
    label_path =  base + '/true_labels/' + filename[:4] + '.mat'

    session_data = loadmat(data_path, squeeze_me=False)
    label_data   = loadmat(label_path, squeeze_me=False) 
    
    gdf_data_path  =  base + '/' + filename[:4] + '.gdf'
    raw_gdf = mne.io.read_raw_gdf(gdf_data_path, stim_channel="auto")
    raw_gdf.load_data()
    
    # Parse data
    s = raw_gdf.get_data().T # cnt -> tnc
    assert np.allclose(s * 1e6, session_data["s"]), (
        "mne and loadmat loaded different singal...")
    h = session_data["h"] # header
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

    assert filename[:4] == origin_filename[:4]
    if verbose:
        print("- filename:", filename)
        print("- load data from:", data_path)
        print('\t- original fileanme:', origin_filename)
        print("- load label from:", label_path)
        print("- shape of s", s.shape) # (time, 25 channels), 
        print("- shape of labels", labels.shape) # (288 trials)

    data =  {"s":s, "h":h, "labels":labels, 
             "filename":filename, "artifacts":artifacts, 
             "typ":typ, "pos":pos, "fs":fs, "dur":dur, 
             "typ2desc":typ2desc, "ch_names":ch_names}
    return data

@verbose_func_name
def s_to_cnt(data, verbose=False):
    data = deepcopy(data)
    assert ("s" in data.keys()) and ("cnt" not in data.keys())
    data["cnt"] = data.pop("s").T
    
    if verbose:
        print("- shape of cnt:", data["cnt"].shape)
    return data

@verbose_func_name
def rerange_label_from_0(data, verbose=False):
    data = deepcopy(data)
    data["labels"] = data["labels"] - 1
    assert np.array_equal(np.unique(data["labels"]), [0,1,2,3])
    
    if verbose:
        print("- unique labels:", np.unique(data["labels"]))
    return data

@verbose_func_name
def rerange_pos_from_0(data, verbose=False):
    """
    In matlab, index starts from 1.
    In python, index starts from 0.
    To adapt index type data, subtract 1 from it.
    """
    data = deepcopy(data)
    data["pos"] = data["pos"] - 1
    assert data["pos"].min() == 0
    
    if verbose:
        print("- initial value:", data["pos"][0])
        print("- minimum value:", np.min(data["pos"]))
    return data


# 2. Preprocessing
#     1. drop EOG channels
#     2. replace break with mean
#     3. scaling (microvolt)
#     4. bandpass 4-38Hz (butterworth 3rd order)
#     5. exponential running standardization (init_block_size=1000, factor_new=1e-3)
#     6. epoch (cue-0.5ms ~ cue+4ms)
#     * no rejection

# In[2]:


import pandas as pd
import numpy as np
from scipy.signal import butter,lfilter
from braindecode.datautil import (
    exponential_moving_standardize  # moving은 최신, running은 예전 꺼. axis가 달라서 중요함!
)

@verbose_func_name
def drop_eog_from_cnt(data, verbose=False):
    assert (
        data["cnt"].shape[0] == 25
        ) and (
        len(data["ch_names"]) == 25
        ), "the number of channels is not 25..."
    data = deepcopy(data)
    data["cnt"] = data["cnt"][0:22]
    data["ch_names"] = data["ch_names"][0:22]
    
    if verbose:
        print("- shape of cnt:", data["cnt"].shape)
    return data

@verbose_func_name
def replace_break_with_mean(data, verbose=False):
    data = deepcopy(data)
    cnt = data["cnt"]
    for i_chan in range(cnt.shape[0]):
        this_chan = cnt[i_chan]
        cnt[i_chan] = np.where(
            this_chan == np.min(this_chan), np.nan, this_chan
        )
        mask = np.isnan(cnt[i_chan])
        chan_mean = np.nanmean(cnt[i_chan])
        cnt[i_chan, mask] = chan_mean
    data["cnt"] = cnt
    assert not np.any(np.isnan(cnt)), "nan remains in cnt.."
    
    if verbose:
        print("- min of cnt:", np.min(cnt))
    return data

@verbose_func_name
def change_scale(data, factor, channels="all", verbose=False):
    """
    Args
    ----
    data : dict
    factor : float
    channels : list of int, or int
    verbose : bool
    """
    data = deepcopy(data)
    if channels == "all":
        channels = list(range(data["cnt"].shape[0]))
    elif isinstance(channels, int):
        channels = [channels]
    
    assert hasattr(channels, "__len__"), (
        "channels should be list or int...")
    
    assert (
        max(channels) <= data["cnt"].shape[0]
    ) and (
        min(channels) >= 0
    ), ("channel index should be between 0 and #channel of data...")
    
    assert ("s" not in data.keys()) and ("cnt" in data.keys())
    
    data["cnt"][channels, :] = data["cnt"][channels, :] * factor
    
    if verbose:
        print("- applied channels:", channels)
        print("- factor :", factor)
        print("- maximum value:", np.max(data["cnt"]))
        print("- minimum value:", np.min(data["cnt"]))
    return data

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_lowpass(highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype="lowpass")
    return b, a

def butter_highpass(lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype="highpass")
    return b, a
    
@verbose_func_name
def butter_bandpass_filter(data, lowcut=0, highcut=0, order=3, 
                           axis=-1, verbose=False):
    assert (lowcut != 0) or (highcut != 0), (
        "one of lowcut and highcut should be not 0..."
    )
    data = deepcopy(data)
    fs = data["fs"]
    
    if lowcut == 0:
        print("banpass changes into lowpass "
              "since lowcut is 0 ...")
        b, a = butter_lowpass(highcut, fs, order)
    elif highcut == 0:
        print("bandpass changes into highpass "
              "since highcut is 0 ...")
        b, a = butter_highpass(lowcut, fs, order)
    else:
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        
    data["cnt"] = lfilter(b, a, data["cnt"], axis=axis)
    if verbose:
        if lowcut == 0:
            print(f"- lowpass : {highcut}Hz")
        elif highcut == 0:
            print(f"- highpass : {lowcut}Hz")
        else:
            print(f"- {lowcut}-{highcut}Hz")
        print(f"- order {order}")
        print(f"- fs {fs}Hz")
    return data
    
@verbose_func_name
def exponential_moving_standardize_from_braindecode(data, 
                         factor_new, 
                         init_block_size, 
                         eps=1e-4, 
                         verbose=False):
    """
    for latest braindecode version...
    exponential_moving_standardize takes cnt (time, channel)
    """
    data = deepcopy(data)
    before_mean = np.mean(data["cnt"], axis=1)
    data["cnt"] = exponential_moving_standardize(
        data["cnt"], 
        factor_new=factor_new, 
        init_block_size=init_block_size,
        eps=eps
    )
    assert np.all(before_mean != np.mean(data["cnt"], axis=1))
    if verbose:
        print("- factor_new", factor_new)
        print("- init_block_size", init_block_size)
        print("- mean before standarization")
        print(before_mean)
        print("- mean after  standarization")
        print(np.mean(data["cnt"], axis=1))
    return data

# @verbose_func_name
# def exponential_running_standardize_from_braindecode(data, 
#                          factor_new, 
#                          init_block_size, 
#                          eps=1e-4, 
#                          verbose=False):
#     """
#     for outdated braindecode version...
#     exponential_running_standardize takes tnc (time, channel)
#     """
#     data = deepcopy(data)
#     before_mean = np.mean(data["cnt"], axis=1)
#     data["cnt"] = exponential_running_standardize(
#         data["cnt"].T, 
#         factor_new=factor_new, 
#         init_block_size=init_block_size,
#         eps=eps
#     ).T
#     assert np.all(before_mean != np.mean(data["cnt"], axis=1))
#     if verbose:
#         print("- factor_new", factor_new)
#         print("- init_block_size", init_block_size)
#         print("- mean before standarization")
#         print(before_mean)
#         print("- mean after  standarization")
#         print(np.mean(data["cnt"], axis=1))
#     return data

@verbose_func_name
def epoch_X_y_from_data(data, 
                           start_sec_offset, 
                           stop_sec_offset, 
                           verbose=False):
    """
    Args
    ----
    data : dict
        It can be obtained by load_gdf2mat and s_to_cnt functions.
    start_sec_offset : int
    stop_sec_offset : int
    verbose : bool
    
    Return
    ------
    X : 3d array (n_trials, n_channels, time)
    y : 2d array (n_trials, 1)
    
    NOTE
    ----
    The base of offset is 'start of a trial onset'.
    NOT based on 'cue onset'. if you want to use offset 
    based on 'cue onset', add 2 sec to start_sec_offset 
    and stop_sec_offset.
    """
    cnt = data["cnt"]
    pos = data["pos"]
    typ = data["typ"]
    fs  = data["fs"]
    
    start_onset = pos[typ == 768] # start of a trial
    trials = []
    for i, onset in enumerate(start_onset):
        trials.append(
            cnt[0:22, 
                int(onset+start_sec_offset*fs):int(onset+stop_sec_offset*fs)] # start of a trial + 1.5 ~ 6
        )
    X = np.array(trials) # trials, channels, time
    y = data["labels"]
    
    if verbose:
        print("- From : start of a trial onset +", start_sec_offset, "sec")
        print("- To   : start of a trial onset +", stop_sec_offset, "sec")
        print("- shape of X", X.shape)
        print("- shape of y", y.shape)
    
    return X, y


# 3. Split
#     - split train into train and validation (8:2)

# In[3]:


@verbose_func_name
def split_train_val(X, y, val_ratio, verbose=False):
    assert (val_ratio < 1) and (val_ratio > 0),            "val_raion not in (0, 1)"
    val_size = round(len(y) * val_ratio)
    X_tr, y_tr   = X[:-val_size], y[:-val_size]
    X_val, y_val = X[-val_size:], y[-val_size:]
    assert (len(X_tr) == len(y_tr)) and (len(X_val) == len(y_val)),(
            "each pair of X and y should have same number of trials...")
    assert len(X) == len(X_tr)+len(X_val),(
            "sum of number of splited trials should equal number of unsplited trials")
    if verbose:
        print("- shape of X_tr", X_tr.shape)
        print("- shape of y_tr", y_tr.shape)
        print("- shape of X_val", X_val.shape)
        print("- shape of y_val", y_val.shape)
    return X_tr, y_tr, X_val, y_val


# 4. Crop (bunch of crops)
#     - input_time_length: 1000 samples
#     * augmentation effect (twice)

# In[4]:


import torch

class CropDataset:
    def __init__(self, X, y, input_time_length, verbose=False):
        self.X, self.y, self.trial_inds = bunch_of_crops(
            X, y, input_time_length=input_time_length, verbose=verbose
        )
        
    def __getitem__(self, crop_ind):
        return self.X[crop_ind], self.y[crop_ind], self.trial_inds[crop_ind]
    
    def __len__(self):
        return len(self.y)

@verbose_func_name
def bunch_of_crops(X, y, input_time_length, verbose=False):
    """
    Args
    ----
    X : 4d Tensor (n_trials, 1, n_channels, n_time_samples)
    y : 2d Tensor (n_trials, 1)
    input_time_length : int
    """
    assert len(X) == len(y)
    assert X.shape[3] > input_time_length
    new_X = []
    new_y = []
    trial_inds = []
    for i_trial, (this_x, this_y) in enumerate(zip(X, y)):
        new_X.append(this_x[:, :, :input_time_length])
        new_y.append(this_y)
        trial_inds.append(i_trial)
        
        new_X.append(this_x[:, :, -input_time_length:])
        new_y.append(this_y)
        trial_inds.append(i_trial)
        
    new_X = torch.stack(new_X, dim=0)
    new_y = torch.stack(new_y, dim=0)
    trial_inds = torch.Tensor(trial_inds)
    assert (
        len(new_X) == len(new_y)
    ) and (
        len(new_y) == len(trial_inds)
    ), "length of new_X, new_y and trial_inds should be same..."
    if verbose:
        print("- shape of new_X", new_X.shape)
        print("- shape of new_y", new_y.shape)
        print("- shape of trial_inds", trial_inds.shape)
    return new_X, new_y, trial_inds
    
class TrialDataset:
    def __init__(self, X, y, verbose=False):
        assert len(X) == len(y), "X and y should have same length..."
        self.X = X
        self.y = y
        self.trial_inds = torch.arange(len(y))
        
        if verbose:
            print("\nTrials")
            print("- shape of X", self.X.shape)
            print("- shape of y", self.y.shape)
            print("- shape of trial_inds", self.trial_inds.shape)
        
    def __getitem__(self, cur_ind):
        return self.X[cur_ind], self.y[cur_ind], self.trial_inds[cur_ind]
    
    def __len__(self):
        return len(self.y)


# 5. Data loader

# In[5]:


from torch.utils.data import DataLoader


# 6. Model
#     - ShallowNet
#     - to_dense_prediction_model

# In[6]:


from torch import nn

class Square(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x ** 2

class Log(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.log(x)
    
class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x = torch.squeeze(x, dim=3)
        return torch.squeeze(x, dim=2)

class TransposeTimeChannel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x.permute(0,1,3,2)


# In[7]:


from torch import nn
from torch.nn import init
from collections import OrderedDict
from braindecode.models.util import to_dense_prediction_model

@verbose_func_name
def make_shallownet_cropped(verbose=False):
    model = nn.Sequential(OrderedDict([
        ("transpose_time_channel", TransposeTimeChannel()),
        ("temp_conv", nn.Conv2d(1,  40, kernel_size=(25,1), stride=(1,1))),
        ("spat_conv", nn.Conv2d(40, 40, kernel_size=(1,22), stride=(1,1), bias=False)),
        ("bn", nn.BatchNorm2d(40)),
        ("square", Square()),
        ("mean_pool", nn.AvgPool2d(kernel_size=(75,1), stride=(15,1))),
        ("log", Log()),
        ("dropout", nn.Dropout(0.5)), # nn.Dropout(0.5) is used in braindecode...
        ("clf_conv", nn.Conv2d(40,4, kernel_size=(30,1), stride=(1,1))),
        ("squeeze", Squeeze())
    ]))

    # xavier initialization is used braindecode...
    # Initialization, xavier is same as in paper...
    init.xavier_uniform_(model.temp_conv.weight, gain=1)
    init.constant_(model.temp_conv.bias, 0)

    init.xavier_uniform_(model.spat_conv.weight, gain=1)

    init.constant_(model.bn.weight, 1)
    init.constant_(model.bn.bias, 0)

    init.xavier_uniform_(model.clf_conv.weight, gain=1)
    init.constant_(model.clf_conv.bias, 0)
    
    if verbose:
        print(model)
    
    return model

@verbose_func_name
def make_deepnet_cropped(verbose=True):
    model = nn.Sequential(OrderedDict([
        ("transpose_time_channel", TransposeTimeChannel()),
        # Conv Pool Block 1
        ("temp_conv", nn.Conv2d(1,  25, kernel_size=(10,1), stride=(1,1))),
        ("spat_conv", nn.Conv2d(25, 25, kernel_size=(1,22), stride=(1,1), bias=False)),
        ("bn_0", nn.BatchNorm2d(25)),
        ("elu_0", nn.ELU()),
        ("max_pool_0", nn.MaxPool2d(kernel_size=(3,1), stride=(3,1))),
        # Conv Pool Block 2
        ("dropout_1", nn.Dropout(0.5)),
        ("gen_conv_1", nn.Conv2d(25, 50, kernel_size=(10,1), stride=(1,1), bias=False)), # general convolutional layer
        ("bn_1", nn.BatchNorm2d(50)),
        ("elu_1", nn.ELU()),
        ("max_pool_1", nn.MaxPool2d(kernel_size=(3,1), stride=(3,1))),
        # Conv Pool Block 3
        ("dropout_2", nn.Dropout(0.5)),
        ("gen_conv_2", nn.Conv2d(50, 100, kernel_size=(10,1), stride=(1,1), bias=False)),
        ("bn_2", nn.BatchNorm2d(100)),
        ("elu_2", nn.ELU()),
        ("max_ppol_2", nn.MaxPool2d(kernel_size=(3,1), stride=(3,1))),
        # Conv Pool Block 4
        ("dropout_3", nn.Dropout(0.5)),
        ("gen_conv_3", nn.Conv2d(100, 200, kernel_size=(10,1), stride=(1,1), bias=False)),
        ("bn_3", nn.BatchNorm2d(200)),
        ("elu_3", nn.ELU()),
        ("max_pool_3", nn.MaxPool2d(kernel_size=(3,1), stride=(3,1))),
        # Classification Layer
#         ("dropout_4", nn.Dropout(0.5)),
        ("clf_conv", nn.Conv2d(200, 4, kernel_size=(2,1), stride=(1,1))),
        ("squeeze", Squeeze())
    ]))    
    
    # xavier initialization is used braindecode...
    # Initialization, xavier is same as in paper...
    # Conv Pool Block 1
    init.xavier_uniform_(model.temp_conv.weight, gain=1)
    init.constant_(model.temp_conv.bias, 0)

    init.xavier_uniform_(model.spat_conv.weight, gain=1)

    init.constant_(model.bn_0.weight, 1)
    init.constant_(model.bn_0.bias, 0)
    
    # Conv Pool Block 2
    init.xavier_uniform_(model.gen_conv_1.weight, gain=1)
    
    init.constant_(model.bn_1.weight, 1)
    init.constant_(model.bn_1.bias, 0)
    
    # Conv Pool Block 3
    init.xavier_uniform_(model.gen_conv_2.weight, gain=1)
    
    init.constant_(model.bn_2.weight, 1)
    init.constant_(model.bn_2.bias, 0)
    
    # Conv Pool Block 4
    init.xavier_uniform_(model.gen_conv_3.weight, gain=1)
    
    init.constant_(model.bn_3.weight, 1)
    init.constant_(model.bn_3.bias, 0)
    
    # Clasification Layer
    init.xavier_uniform_(model.clf_conv.weight, gain=1)
    init.constant_(model.clf_conv.bias, 0)
    
    if verbose:
        print(model)
    
    return model

@verbose_func_name
def make_shallownet_trialwise(verbose=True):
    model = nn.Sequential(OrderedDict([
        ("transpose_time_channel", TransposeTimeChannel()),
        ("temp_conv", nn.Conv2d(1,  40, kernel_size=(25,1), stride=(1,1))),
        ("spat_conv", nn.Conv2d(40, 40, kernel_size=(1,22), stride=(1,1), bias=False)),
        ("bn", nn.BatchNorm2d(40)),
        ("square", Square()),
        ("mean_pool", nn.AvgPool2d(kernel_size=(75,1), stride=(15,1))),
        ("log", Log()),
        ("dropout", nn.Dropout(0.5)), # nn.Dropout(0.5) is used in braindecode...
        ("clf_conv", nn.Conv2d(40,4, kernel_size=(69,1), stride=(1,1))),
        ("squeeze", Squeeze())
    ]))

    # xavier initialization is used braindecode...
    # Initialization, xavier is same as in paper...
    init.xavier_uniform_(model.temp_conv.weight, gain=1)
    init.constant_(model.temp_conv.bias, 0)

    init.xavier_uniform_(model.spat_conv.weight, gain=1)

    init.constant_(model.bn.weight, 1)
    init.constant_(model.bn.bias, 0)

    init.xavier_uniform_(model.clf_conv.weight, gain=1)
    init.constant_(model.clf_conv.bias, 0)
    
    if verbose:
        print(model)
    
    return model

@verbose_func_name
def make_deepnet_trialwise(verbose=True):
    model = nn.Sequential(OrderedDict([
        ("transpose_time_channel", TransposeTimeChannel()),
        # Conv Pool Block 1
        ("temp_conv", nn.Conv2d(1,  25, kernel_size=(10,1), stride=(1,1))),
        ("spat_conv", nn.Conv2d(25, 25, kernel_size=(1,22), stride=(1,1), bias=False)),
        ("bn_0", nn.BatchNorm2d(25)),
        ("elu_0", nn.ELU()),
        ("max_pool_0", nn.MaxPool2d(kernel_size=(3,1), stride=(3,1))),
        # Conv Pool Block 2
        ("dropout_1", nn.Dropout(0.5)),
        ("gen_conv_1", nn.Conv2d(25, 50, kernel_size=(10,1), stride=(1,1), bias=False)), # general convolutional layer
        ("bn_1", nn.BatchNorm2d(50)),
        ("elu_1", nn.ELU()),
        ("max_pool_1", nn.MaxPool2d(kernel_size=(3,1), stride=(3,1))),
        # Conv Pool Block 3
        ("dropout_2", nn.Dropout(0.5)),
        ("gen_conv_2", nn.Conv2d(50, 100, kernel_size=(10,1), stride=(1,1), bias=False)),
        ("bn_2", nn.BatchNorm2d(100)),
        ("elu_2", nn.ELU()),
        ("max_ppol_2", nn.MaxPool2d(kernel_size=(3,1), stride=(3,1))),
        # Conv Pool Block 4
        ("dropout_3", nn.Dropout(0.5)),
        ("gen_conv_3", nn.Conv2d(100, 200, kernel_size=(10,1), stride=(1,1), bias=False)),
        ("bn_3", nn.BatchNorm2d(200)),
        ("elu_3", nn.ELU()),
        ("max_pool_3", nn.MaxPool2d(kernel_size=(3,1), stride=(3,1))),
        # Classification Layer
        ("dropout_4", nn.Dropout(0.5)),
        ("clf_conv", nn.Conv2d(200, 4, kernel_size=(9,1), stride=(1,1))),
        ("squeeze", Squeeze())
    ]))    
    
    # xavier initialization is used braindecode...
    # Initialization, xavier is same as in paper...
    # Conv Pool Block 1
    init.xavier_uniform_(model.temp_conv.weight, gain=1)
    init.constant_(model.temp_conv.bias, 0)

    init.xavier_uniform_(model.spat_conv.weight, gain=1)

    init.constant_(model.bn_0.weight, 1)
    init.constant_(model.bn_0.bias, 0)
    
    # Conv Pool Block 2
    init.xavier_uniform_(model.gen_conv_1.weight, gain=1)
    
    init.constant_(model.bn_1.weight, 1)
    init.constant_(model.bn_1.bias, 0)
    
    # Conv Pool Block 3
    init.xavier_uniform_(model.gen_conv_2.weight, gain=1)
    
    init.constant_(model.bn_2.weight, 1)
    init.constant_(model.bn_2.bias, 0)
    
    # Conv Pool Block 4
    init.xavier_uniform_(model.gen_conv_3.weight, gain=1)
    
    init.constant_(model.bn_3.weight, 1)
    init.constant_(model.bn_3.bias, 0)
    
#     # Clasification Layer
    init.xavier_uniform_(model.clf_conv.weight, gain=1)
    init.constant_(model.clf_conv.bias, 0)
    
    if verbose:
        print(model)
    
    return model


# 7. Learning strategy
#     - loss function : log softmax + NLLloss for each crop
#                       ** tied sample loss ? **
#     - optimizer : Adam
#     - evaluater for cropped learning
#     - early stop : 1. using training, no decrease on val acc (80 epoch) or max epoch 800
#                    2. using training and val, same training loss with val loss from first stop or max epoch 800.
#     - maxnorm

# In[8]:


import torch
from torch.functional import F

def crop_ce_loss(outputs, labels):
    """
    Arg
    ---
    outputs : 3d Tensor (n_batch, n_classes, n_preds_per_input)
    labels  : 2d Tensor (n_batch, 1)
    """
    assert outputs.dim() == 3
    assert len(outputs) == len(labels)
    n_batch, n_classes, n_preds_per_input = outputs.shape
    
    out = outputs.permute(1,0,2)
    out = out.reshape(n_classes, n_batch*n_preds_per_input)
    out = out.permute(1,0)
    
    lab = labels * labels.new_ones(n_batch, n_preds_per_input)
    lab = lab.reshape(n_batch*n_preds_per_input)
    return F.cross_entropy(out, lab)

def tied_sample_loss(outputs):
    """
    Arg
    ---
    outputs : 3d Tensor (n_batch, n_classes, n_preds_per_input)
    """
    assert outputs.dim() == 3
    this_prob = F.softmax(outputs[:,:,:-1], dim=1)
    next_prob = F.softmax(outputs[:,:,1:], dim=1)

    loss = torch.sum(-torch.log(this_prob) * next_prob, dim=1) # sum over dim of n_classes
    return torch.mean(loss)

def trial_pred_from_crop_outputs(outputs):
    """
    Args
    ----
    outputs : 3d tensor (n_batch, n_classes, n_preds_per_input)
    
    Return
    ------
    preds : 2d tensor (n_trials, 1)
    """
    assert outputs.dim() == 3
#     probs = torch.softmax(outputs, dim=1).mean(dim=2) # as I understood...
    probs = F.log_softmax(outputs, dim=1).mean(dim=2) # according to github code
    return torch.argmax(probs, dim=1, keepdim=True)
        
def evaluate_with_dataloader(model, trial_dataloaders, mode, 
                             use_tied_loss, which_learning):
    if which_learning == "trialwise":
        return evaluate_trialwise_with_dataloader(
            model, trial_dataloaders, mode
        )
    elif which_learning == "cropped":
        return evaluate_cropped_with_dataloader(
            model, trial_dataloaders, mode, use_tied_loss
        )
    
def evaluate_cropped_with_dataloader(model, trial_dataloaders, mode, use_tied_loss):
    """
    Args
    ----
    model : `:class: torch.nn.Module`
    dataloader : dict of ':class: torch.utils.data.DataLoader'
        "tr", "val" and "te" are in dict.keys()
    mode : str
        "tr", "val" or "te"
    use_tied_loss : bool
    Return
    ------
    results : dict
    """
    model.eval()
    total_outputs = []
    total_labels  = []
    with torch.no_grad():
        for inputs, labels, _ in trial_dataloaders[mode]:
            total_outputs.append(model(inputs))
            total_labels.append(labels)
        
        total_outputs = torch.cat(total_outputs)
        total_labels  = torch.cat(total_labels)
        total_preds = trial_pred_from_crop_outputs(total_outputs)
        
        assert total_preds.shape == total_labels.shape
        total_corrects = total_preds == total_labels
        acc = torch.mean(total_corrects.float())
        ce_loss = crop_ce_loss(total_outputs, total_labels) 
        
        if use_tied_loss:
            tied_loss = tied_sample_loss(total_outputs)
            return {
                    f"{mode}_acc":acc.item(), 
                    f"{mode}_loss":ce_loss.item()+tied_loss.item(), 
                    f"{mode}_ce_loss":ce_loss.item(), 
                    f"{mode}_tied_loss":tied_loss.item()
                   }
        else:
            return {
                    f"{mode}_acc":acc.item(), 
                    f"{mode}_ce_loss":ce_loss.item(), 
                   }

def evaluate_trialwise_with_dataloader(model, trial_dataloaders, mode):
    """
    Args
    ----
    model : `:class: torch.nn.Module`
    dataloader : dict of ':class: torch.utils.data.DataLoader'
        "tr", "val" and "te" are in dict.keys()
    mode : str
        "tr", "val" or "te"
    Return
    ------
    results : dict
    """
    model.eval()
    total_outputs = []
    total_labels  = []
    with torch.no_grad():
        for inputs, labels, _ in trial_dataloaders[mode]:
            total_outputs.append(model(inputs))
            total_labels.append(labels)
        
        total_outputs = torch.cat(total_outputs)
        total_labels  = torch.cat(total_labels)
        total_preds = torch.softmax(total_outputs, dim=1).argmax(dim=1, keepdim=True)
        
        assert total_preds.shape == total_labels.shape
        total_corrects = total_preds == total_labels
        acc = torch.mean(total_corrects.float())
        ce_loss = F.cross_entropy(total_outputs, total_labels.flatten()) 
    
    return {
        f"{mode}_acc":acc.item(), 
        f"{mode}_ce_loss":ce_loss.item(), 
           }

class EarlyStopNoIncrease:
    def __init__(self, column_name, patient_epochs, min_increase=1e-6):
        self.column_name = column_name
        self.patient_epochs = patient_epochs
        self.min_increase = min_increase
        self.best_epoch = 0
        self.best_val = 0

    def __call__(self, epoch_df):
        """
        Args
        ---- 
        epoch_df : pandas.DataFrame
        column_name : str
        patient_epoch : int
        margin_of_error : float, [0,1]

        Return
        ------
        stop : bool
        """
        assert self.column_name in epoch_df.columns, (
            f"{self.column_name} not in epoch_df...")
        this_epoch = epoch_df.index[-1]
        this_val = epoch_df[self.column_name].iloc[-1]
        if this_val > ((1+self.min_increase) * self.best_val):
            self.best_epoch = this_epoch
            self.best_val = this_val
        return (this_epoch - self.best_epoch) >= self.patient_epochs

def early_stop_reach_below(epoch_df, column_name, base_value):
    """
    Args
    ----
    epoch_df : pandas.DataFrame
    column_name : str
    base_value : float
    margin_of_error : float, [0,1]
    """
    assert column_name in epoch_df.columns, (
        f"{column_name} not in epoch_df...")
    return base_value >= epoch_df[column_name].iloc[-1]

class SaveBestModel:
    def __init__(self):
        self.best_val_acc = 0
        self.saved_weight = None
        self.saved_optimizer = None
        self.saved_epoch = -1

    def if_best_val_acc(self, model, optimizer, epoch_df):
        this_val_acc = epoch_df["val_acc"].iloc[-1]
        if self.best_val_acc <= this_val_acc:
            self.best_val_acc = this_val_acc
            self.saved_weight = model.state_dict().copy()
            self.saved_optimizer = optimizer.state_dict().copy()
            self.saved_epoch = epoch_df.index[-1]
            print("  new best val acc:", this_val_acc)
            
    def restore_best_model(self, model, optimizer, epoch_df):
        print("  model load weight saved at", self.saved_epoch, "epoch")
        model.load_state_dict(self.saved_weight.copy())
        optimizer.load_state_dict(self.saved_optimizer.copy())
        epoch_df.drop(range(self.saved_epoch + 1, len(epoch_df)), inplace=True)


# In[9]:


def maxnorm(model):
    last_weight = None
    for name, module in list(model.named_children()):
        if hasattr(module, "weight") and (
            not module.__class__.__name__.startswith("BatchNorm")
        ):
            module.weight.data = torch.renorm(
                module.weight.data, 2, 0, maxnorm=2
            )
            last_weight = module.weight
    if last_weight is not None:
        last_weight.data = torch.renorm(last_weight.data, 2, 0, maxnorm=0.5)


# ---

# ---

# In[ ]:


import argparse
import os
import logging
import sys

if __name__ == '__main__':
    def str2bool(v): 
        if isinstance(v, bool): 
            return v 
        if v.lower() in ('yes', 'true', 't', 'y', '1'): 
            return True 
        elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
            return False 
        else: raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, type=int)
    parser.add_argument("--lowcut", required=True, type=int,
                       help="0 or 4")
    parser.add_argument("--which_model", required=True,
                       help="ShallowNet or DenseNet")
    parser.add_argument("--which_learning", required=True,
                        help="trialwise or cropped")
    parser.add_argument("--use_tied_loss", required=True, type=str2bool,
                       help="True or False")
    parser.add_argument("--device", default="cuda",
                       help="cuda or cpu")
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--repeat", default=1, type=int)
    args = parser.parse_args()
    
    assert os.path.exists(args.result_dir)

    log = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)
    def print(*texts):
        texts = " ".join([str(_) for _ in texts])
        for line in texts.split("\n"):
            log.info(line)
        
    print("args:", args)


    # In[ ]:


    subject = args.subject
    # data_dir = "/home/jinhyo/multi_class_motor_imagery/data/BCICIV_2a/gdf"
    data_dir = "/home/jinhyo/JHS_server2/multi_class_motor_imagery/data/BCICIV_2a/gdf"

    lowcut = args.lowcut
    highcut = 38 
    order = 3
    factor_new = 1e-3
    init_block_size = 1000
    start_sec_offset=1.5
    stop_sec_offset=6.0

    input_time_length = 1000
    which_model = args.which_model
    which_learning = args.which_learning
    device = args.device

    batch_size = 60
    max_epochs = 800
    patient_epochs = 80
    use_tied_loss = args.use_tied_loss

    assert lowcut in [0, 4]
    assert which_model in ["ShallowNet", "DeepNet"]
    assert which_learning in ["trialwise", "cropped"]
    assert isinstance(use_tied_loss, bool)

    if (which_learning == "trialwise") and use_tied_loss:
        raise AttributeError(
            "trialwise learning is not compatible with tied_loss")



    # 1. Data load  & 2. Preprocessing

    # In[12]:


    print("\nLoad Train Data")
    data_tr = load_gdf2mat_feat_mne(subject  = subject, 
                           train             = True, 
                           data_dir          = data_dir,
                           overflowdetection = False,
                           verbose           = True)
    data_tr = s_to_cnt(data_tr, verbose=True)
    data_tr = rerange_label_from_0(data_tr, verbose=True)
    data_tr = rerange_pos_from_0(data_tr, verbose=True)

    print("\nPreprocessing")
    data_tr = drop_eog_from_cnt(data_tr, verbose=True)
    data_tr = replace_break_with_mean(data_tr, verbose=True)
    data_tr = change_scale(data_tr, factor=1e06, 
                           channels="all", verbose=True)
    data_tr = butter_bandpass_filter(data_tr, 
                                     lowcut=lowcut, highcut=highcut, 
                                     order=order, verbose=True)
    data_tr = exponential_moving_standardize_from_braindecode(data_tr, 
                                                               factor_new=factor_new, 
                                                               init_block_size=init_block_size, 
                                                               verbose=True)
    X_tr, y_tr = epoch_X_y_from_data(data_tr, 
                                        start_sec_offset=start_sec_offset, 
                                        stop_sec_offset=stop_sec_offset,
                                        verbose=True)


    # In[13]:


    print("\nLoad Test Data")
    data_te = load_gdf2mat_feat_mne(subject  = subject, 
                           train             = False, 
                           data_dir          = data_dir,
                           overflowdetection = False,
                           verbose           = True)
    data_te = s_to_cnt(data_te, verbose=True)
    data_te = rerange_label_from_0(data_te, verbose=True)
    data_te = rerange_pos_from_0(data_te, verbose=True)

    print("\nPreprocessing")
    data_te = drop_eog_from_cnt(data_te, verbose=True)
    data_te = replace_break_with_mean(data_te, verbose=True)
    data_te = change_scale(data_te, factor=1e06, 
                           channels="all", verbose=True)
    data_te = butter_bandpass_filter(data_te, 
                                     lowcut=lowcut, highcut=highcut, 
                                     order=order, verbose=True)
    data_te = exponential_moving_standardize_from_braindecode(data_te, 
                                                               factor_new=factor_new, 
                                                               init_block_size=init_block_size, 
                                                               verbose=True)
    X_te, y_te = epoch_X_y_from_data(data_te, 
                                        start_sec_offset=start_sec_offset, 
                                        stop_sec_offset=stop_sec_offset,
                                        verbose=True)


    # 3. Split

    # In[14]:


    X_tr, y_tr, X_val, y_val = split_train_val(X_tr, y_tr, 
                                               val_ratio=0.2, 
                                               verbose=True)


    # 4. Crop (bunch of crops)

    # In[15]:


    print("\nTo tensor")
    # channel first
    X_tr  = torch.Tensor(X_tr[:,None,:,:]).to(device)
    y_tr  = torch.Tensor(y_tr).long().to(device)
    print("- shape of X_tr:", X_tr.shape)
    print("- shape of y_tr:", y_tr.shape)

    X_val = torch.Tensor(X_val[:,None,:,:]).to(device)
    y_val = torch.Tensor(y_val).long().to(device)
    print("- shape of X_val:", X_val.shape)
    print("- shape of y_val:", y_val.shape)

    X_te  = torch.Tensor(X_te[:,None,:,:]).to(device)
    y_te  = torch.Tensor(y_te).long().to(device)
    print("- shape of X_te:", X_te.shape)
    print("- shape of y_te:", y_te.shape)


    # In[16]:


    print("\nDataset")
    if which_learning == "trialwise":
        print("\ntrain + val")
        trial_dataset_tr_val = TrialDataset(torch.cat([X_tr, X_val]),
                                          torch.cat([y_tr, y_val]), 
                                          verbose=True)
    elif which_learning == "cropped":
        print("\ntrain")
        crop_dataset_tr  = CropDataset(X_tr,  y_tr,  
                                       input_time_length=input_time_length, 
                                       verbose=True)
        print("\ntrain + val")
        crop_dataset_tr_val = CropDataset(torch.cat([X_tr, X_val]),
                                          torch.cat([y_tr, y_val]), 
                                          input_time_length=input_time_length, 
                                          verbose=True)
    print("\ntrain")
    trial_dataset_tr  = TrialDataset(X_tr,  y_tr,
                                   verbose=True)
    print("\nval")
    trial_dataset_val = TrialDataset(X_val, y_val,
                                   verbose=True)
    print("\ntest")
    trial_dataset_te  = TrialDataset(X_te,  y_te,
                                   verbose=True)


    # 5. Data loader

    # In[17]:


    if which_learning == "trialwise":
        # for first training
        first_training_dataloader = DataLoader(trial_dataset_tr,
                                         batch_size=batch_size,
                                         shuffle=True)
        # for second training
        second_training_dataloader = DataLoader(trial_dataset_tr_val,
                                             batch_size=batch_size,
                                             shuffle=True)

    elif which_learning == "cropped":
        # for first training
        first_training_dataloader = DataLoader(crop_dataset_tr,
                                         batch_size=batch_size,
                                         shuffle=True)
        # for second training
        second_training_dataloader = DataLoader(crop_dataset_tr_val,
                                             batch_size=batch_size,
                                             shuffle=True)

    # for evaluation
    trial_dataloaders = {"tr" : DataLoader(trial_dataset_tr,
                                            batch_size=batch_size,
                                            shuffle=False),
                          "val" : DataLoader(trial_dataset_val,
                                             batch_size=batch_size,
                                             shuffle=False),
                          "te" : DataLoader(trial_dataset_te,
                                            batch_size=batch_size,
                                            shuffle=False)}


    results = []
    for i in range(1,1+args.repeat):
        print("\n# TRY", i, "\n")
        # 6. Model

        # In[18]:


        if which_learning == "trialwise":
            if which_model == "ShallowNet":
                model = make_shallownet_trialwise(verbose=True)
            elif which_model == "DeepNet":
                model = make_deepnet_trialwise(verbose=True)

        elif which_learning == "cropped":
            if which_model == "ShallowNet":
                model = make_shallownet_cropped(verbose=True)
            elif which_model == "DeepNet":
                model = make_deepnet_cropped(verbose=True)
            to_dense_prediction_model(model)
            print("\nto_dense_prediction_model")
            print(model)

        model = model.to(device)


        # In[19]:


        # for dummy_input, dummy_label, _ in first_training_dataloader:
        #     print("shape of input", dummy_input.shape)
        #     dummy_output = model(dummy_input)
        #     print("shape of output", dummy_output.shape)
        #     break


        # n_params for cropped :  
        # shallownet 41124  
        # deepnet 278879

        # 7. Learning strategy

        # In[20]:


        class LossCase:
            def __init__(self, which_learning, use_tied_loss):
                self.which_learning = which_learning
                self.use_tied_loss = use_tied_loss

            def __call__(self, outputs, labels):
                if self.which_learning == "trialwise":
                    return F.cross_entropy(outputs, labels.flatten())

                elif self.which_learning == "cropped":
                    if self.use_tied_loss:
                        return crop_ce_loss(outputs, labels) + tied_sample_loss(outputs)
                    else :
                        return crop_ce_loss(outputs, labels)


        # In[21]:


        optimizer = torch.optim.Adam(model.parameters())
        save_best_model = SaveBestModel()
        early_stop_no_increase = EarlyStopNoIncrease(column_name="val_acc", 
                                                     patient_epochs=patient_epochs)
        loss_function = LossCase(which_learning=which_learning,
                                 use_tied_loss=use_tied_loss)

        print("\nFirst Training")
        epoch_df = pd.DataFrame()
        for epoch in range(0, max_epochs):
            # Train
            model.train()
            for inputs, labels, _ in first_training_dataloader:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                maxnorm(model)
            # Evaluation
            evaluation_tr  = evaluate_with_dataloader(model, 
                                             trial_dataloaders=trial_dataloaders, 
                                             mode="tr",
                                             use_tied_loss=use_tied_loss,
                                             which_learning=which_learning)
            evaluation_val = evaluate_with_dataloader(model, 
                                             trial_dataloaders=trial_dataloaders, 
                                             mode="val",
                                             use_tied_loss=use_tied_loss,
                                             which_learning=which_learning)
            evaluation_te  = evaluate_with_dataloader(model, 
                                             trial_dataloaders=trial_dataloaders, 
                                             mode="te",
                                             use_tied_loss=use_tied_loss,
                                             which_learning=which_learning)

            assert len(epoch_df) == epoch
            epoch_df = epoch_df.append(
                dict(**evaluation_tr, **evaluation_val, **evaluation_te),
                ignore_index=True
            )
            print("epoch", epoch)
            print(epoch_df.iloc[-1])
            print()

            save_best_model.if_best_val_acc(model, optimizer, epoch_df)

            # Early stop
            if early_stop_no_increase(epoch_df):
                print("\nFirst Early Stop")
                print("- epoch:", epoch)
                print("- restore best model from epoch", save_best_model.saved_epoch)
                save_best_model.restore_best_model(model, optimizer, epoch_df)
                loss_to_reach = epoch_df["tr_ce_loss"].iloc[-1]
                print("- loss to reach:", loss_to_reach)
                break


        # In[22]:


        print("\nSecond Training")
        stop_column_name = "val_loss" if use_tied_loss else "val_ce_loss"
        print("- stop_column_name:", stop_column_name)
        for epoch in range(save_best_model.saved_epoch + 1, 
                           save_best_model.saved_epoch * 2):
            # Train
            model.train()
            for inputs, labels, _ in second_training_dataloader:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                maxnorm(model)

            # Evaluation
            evaluation_tr  = evaluate_with_dataloader(model, 
                                                     trial_dataloaders=trial_dataloaders, 
                                                     mode="tr",
                                                     use_tied_loss=use_tied_loss,
                                                     which_learning=which_learning)
            evaluation_val = evaluate_with_dataloader(model, 
                                                     trial_dataloaders=trial_dataloaders, 
                                                     mode="val",
                                                     use_tied_loss=use_tied_loss,
                                                     which_learning=which_learning)
            evaluation_te  = evaluate_with_dataloader(model, 
                                                     trial_dataloaders=trial_dataloaders, 
                                                     mode="te",
                                                     use_tied_loss=use_tied_loss,
                                                     which_learning=which_learning)

            assert len(epoch_df) == epoch
            epoch_df = epoch_df.append(dict(**evaluation_tr, **evaluation_val, **evaluation_te),
                            ignore_index=True)
            print("epoch", epoch)
            print(epoch_df.iloc[-1])
            print()

            # Early stop
            if early_stop_reach_below(epoch_df, 
                                      column_name=stop_column_name,
                                      base_value=loss_to_reach):
                print("\nSecond Early Stop")
                break
        # else:
        #     print("Couldn't reach to loss", loss_to_reach)
        #     print("current loss", epoch_df[stop_column_name].iloc[-1])
        #     print("Restore best model from first training")
        #     save_best_model.restore_best_model(model, optimizer, epoch_df)


        # In[23]:


        print("\nLast Epoch")
        print(epoch_df.iloc[-1])


        # In[24]:


        # epoch_df[["tr_acc", "val_acc", "te_acc"]].plot()


        # In[25]:


        # epoch_df[["tr_loss", "val_loss", "te_loss"]].plot()


        # In[26]:


        # epoch_df[["tr_ce_loss", "val_ce_loss", "te_ce_loss"]].plot()


        # In[27]:


        result_name = f"{args.result_dir}/{which_model}_subject{subject}_try{i}"
        epoch_df.to_csv(result_name+".csv")
        torch.save(model.state_dict(), result_name+".h5")


        # In[ ]:
        results.append(round(epoch_df["te_acc"].iloc[-1], 2))
        
    print(f"\n{args.repeat} results (te_acc)")
    for i in range(1, 1+args.repeat):
        print(f"{i} try : {results[i-1]}")
    print(f"mean : {np.mean(results):.2f} ± {np.std(results):.2f}")



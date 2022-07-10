#!/usr/bin/env python
# coding: utf-8

# In[1]:
# data.py

import numpy as np
import torch
from collections import OrderedDict
import logging
import sys
sys.path += ["/home/jinhyo/JHS_server1/multi_class_motor_imagery"]
from bcitools.bcitools import (
    verbose_func_name,
    # load_gdf2mat_feat_mne,
    s_to_cnt,
    rerange_label_from_0,
    # rerange_pos_from_0,
    # drop_eog_from_cnt,
    # replace_break_with_mean,
    # change_scale,
    butter_bandpass_filter,
    exponential_moving_standardize_from_braindecode,
    # epoch_X_y_from_data
)
from braindecode.datasets.bbci import BBCIDataset
from copy import deepcopy
import resampy
import torch.distributed as dist
import torch.multiprocessing as mp
import datetime

if __name__ == "__main__":
    log = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s : %(message)s",
        level=logging.DEBUG,
        stream=sys.stdout,
    )

    def print(*texts):
        texts = " ".join([str(_) for _ in texts])
        for line in texts.split("\n"):
            log.info(line)


@verbose_func_name
def epoch_X_y_from_data_HGD(data, start_sec_offset, stop_sec_offset, verbose=False):
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
    fs = data["fs"]

    cue_mask = list(map(lambda x:x in ["Right Hand", "Left Hand", "Rest", "Feet"], typ))
    cue_onset = pos[cue_mask]
    trials = []
    for i, onset in enumerate(cue_onset):
        trials.append(
            cnt[
                :,
                int(onset + start_sec_offset * fs) : int(onset + stop_sec_offset * fs),
            ]  # start of a trial + 1.5 ~ 6
        )
    X = np.array(trials)  # trials, channels, time
    y = data["labels"]
    assert len(X) == len(y)
    
    if verbose:
        print("- From : start of a trial onset +", start_sec_offset, "sec")
        print("- To   : start of a trial onset +", stop_sec_offset, "sec")
        print("- shape of X", X.shape)
        print("- shape of y", y.shape)

    return X, y


@verbose_func_name
def load_HGD(subject, train, data_dir=".", verbose=False):
    """
    reference code
    --------------
    https://gin.g-node.org/robintibor/high-gamma-dataset/src/master/example.py
    """
    train_test = "train" if train is True else "test"
    filename = "{}/{}.mat".format(train_test, subject)
    file_path = "{}/{}".format(data_dir, filename)
    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    loader = BBCIDataset(file_path)

    if verbose:
        print("Loading data...")
    raw_gdf = loader.load()
    
    # parse data
    s = raw_gdf.get_data().T # cnt -> s(tnc)
    fs = raw_gdf.info["sfreq"] # 500.0
    pos = (raw_gdf.annotations.onset * fs).round().astype(int)
    typ = raw_gdf.annotations.description
    ch_names = raw_gdf.ch_names
    labels = raw_gdf.info["events"][:,2].reshape(-1,1) # equivalent with typ
    typ2desc = {
        "Right Hand": "Cue onset Right Hand (class 1)",
        "Left Hand": "Cue onset Left Hand (class 2)",
        "Rest": "Cue onset Rest (class 3)",
        "Feet": "Cue onset Feet (class 4)",
    }

    if verbose:
        print("- folder/filename:", filename)
        print("- load data from:", file_path)
        print("- shape of s", s.shape)  # (time, 129 channels),
        print("- shape of labels", labels.shape)  # (trials,)

    data = {
        "s":s,
        "labels":labels,
        "pos":pos,
        "ch_names":ch_names,
        "fs":fs,
        "typ":typ,
        "typ2desc":typ2desc,
    }
    return data
        
        
@verbose_func_name
def clean_trials(data, verbose=False):
    # Cleaning: First find all trials that have absolute microvolt values
    # larger than +- 800 inside them and remember them for removal later
    temp = deepcopy(data)
    X_for_cleaning, _ = epoch_X_y_from_data_HGD(temp, start_sec_offset=0, stop_sec_offset=4)
    clean_trial_mask = np.max(np.abs(X_for_cleaning), axis=(1,2)) < 800

    if verbose:
        print("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
            np.sum(clean_trial_mask),
            len(X_for_cleaning),
            np.mean(clean_trial_mask) * 100))
    
    data = deepcopy(data)
    assert len(data["labels"]) == len(clean_trial_mask)
    assert len(data["pos"]) == len(clean_trial_mask)
    assert len(data["typ"]) == len(clean_trial_mask)
    data["labels"] = data["labels"][clean_trial_mask]
    data["pos"] = data["pos"][clean_trial_mask]
    data["typ"] = data["typ"][clean_trial_mask]
    return data
        

@verbose_func_name
def leave_44ch_from_cnt(data, verbose=False):
    # now pick only sensors with C in their name
    # as they cover motor cortex
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    C_mask = [ind for ind, C_name in enumerate(data["ch_names"]) if C_name in C_sensors]
    
    data = deepcopy(data)
    data["cnt"] = data["cnt"][C_mask,:]
    data["ch_names"] = C_sensors
    assert data["cnt"].shape[0] == len(C_sensors)
    if verbose:
        print("- shape of cnt:", data["cnt"].shape)
    return data

@verbose_func_name
def sort_44ch_from_cnt(data, verbose=False):
    assert len(data["ch_names"]) == 44
    assert data["cnt"].shape[0] == 44
    sorted_ch_names = [
        "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h", "FFC6h",
        "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
        "FCC5h", "FCC3h", "FCC1h", "FCC2h", "FCC4h", "FCC6h",
        "C5", "C3", "C1", "C2", "C4", "C6",
        "CCP5h", "CCP3h", "CCP1h", "CCP2h", "CCP4h", "CCP6h",
        "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
        "CPP5h", "CPP3h", "CPP1h", "CPP2h", "CPP4h", "CPP6h"
    ]
    assert set(sorted_ch_names) == set(data["ch_names"])
    assert len(sorted_ch_names) == 44
    sorted_ch_index = [data["ch_names"].index(ch_name) for ch_name in sorted_ch_names]
    data = deepcopy(data)
    data["ch_names"] = list(np.array(data["ch_names"])[sorted_ch_index])
    data["cnt"] = data["cnt"][sorted_ch_index, :]
    if verbose: 
        print("- sorted_ch_names:", data["ch_names"])
        print("- shape of cnt:", data["cnt"].shape)
    return data
    
@verbose_func_name
def resample_from_cnt(data, fs, verbose=False):
    """
    reference code
    --------------
    https://github.com/robintibor/braindecode/blob/62c9163b29903751a1dff08e243fcfa0bf7a7118/braindecode/mne_ext/signalproc.py#L34
    """
    # Further preprocessings as descibed in paper
    data = deepcopy(data)
    cnt_old = data["cnt"]
    fs_old = data["fs"]
    pos_old = data["pos"]
    
    cnt = resampy.resample(
        cnt_old, sr_orig=fs_old, sr_new=fs, axis=1, filter="kaiser_fast"
    )
    
    data["cnt"] = cnt
    data["fs"] = fs
    data["pos"] = (pos_old / fs_old * fs).round().astype(int)
    return data


def load_and_preprocessing_for_input_band_dicts_exp442(args, train=True):
    order = args.order
    factor_new = args.factor_new
    init_block_size = args.init_block_size
    input_band_dicts = args.input_band_dicts
    subject = args.subject
    data_dir = args.data_dir
    start_sec_offset = args.start_sec_offset
    stop_sec_offset = args.stop_sec_offset
    fs_new = args.fs_new
    
    Xs, ys = OrderedDict(), OrderedDict()
    train_test = "Train" if train is True else "Test"
    for input_band_name, input_band_dict in input_band_dicts.items():
        print(f"\nLoad {train_test} Data for {input_band_name}")
        data_sub = load_HGD(
            subject=subject,
            train=train,
            data_dir=data_dir,
            verbose=True,
        )
        data_sub = s_to_cnt(data_sub, verbose=True)
        data_sub = rerange_label_from_0(data_sub, verbose=True)
        # data_sub = rerange_pos_from_0(data_sub, verbose=True)

        print(f"\nPreprocessing for {input_band_name}")
        data_sub = clean_trials(data_sub, verbose=True)
        # data_sub = drop_eog_from_cnt(data_sub, verbose=True)
        data_sub = leave_44ch_from_cnt(data_sub, verbose=True)
        data_sub = sort_44ch_from_cnt(data_sub, verbose=True)
        data_sub = resample_from_cnt(data_sub, fs=fs_new, verbose=True)
        # data_sub = replace_break_with_mean(data_sub, verbose=True)
        # data_sub = change_scale(data_sub, factor=1e06, channels="all", verbose=True)
        data_sub = butter_bandpass_filter(
            data_sub, lowcut=input_band_dict["lowcut"], highcut=input_band_dict["highcut"], order=order, verbose=True
        )
        data_sub = exponential_moving_standardize_from_braindecode(
            data_sub, factor_new=factor_new, init_block_size=init_block_size, verbose=True
        )
        X_sub, y_sub = epoch_X_y_from_data_HGD(
            data_sub,
            start_sec_offset=start_sec_offset,
            stop_sec_offset=stop_sec_offset,
            verbose=True,
        )
        
        Xs[input_band_name] = X_sub
        ys[input_band_name] = y_sub
        
        if len(Xs) > 1:
            assert not np.array_equal(
                list(Xs.values())[-2], 
                list(Xs.values())[-1]
            )
            assert np.array_equal(
                list(ys.values())[-2], 
                list(ys.values())[-1]
            )
            
    return Xs, ys


def to_tensor(Xs, ys, device):
    print("To tensor")
    for i, input_band_name in enumerate(Xs.keys()):
        print(f"- {input_band_name}")
        Xs[input_band_name] = torch.Tensor(
            Xs[input_band_name][:, None, :, :] # channel first
        ).to(device)
        
        ys[input_band_name] = torch.Tensor(
            ys[input_band_name]
        ).long().to(device)
        
        print("- shape of X_sub:", Xs[input_band_name].shape)
        print("- shape of y_sub:", ys[input_band_name].shape)
    return Xs, ys

    
class MultiDictDataset:
    def __init__(self, Xs, ys, verbose=False):
        assert len(Xs) == len(ys), "Xs and ys should have same length..."
        Xs_values = list(Xs.values())
        X_first = Xs_values[0]
        for X in Xs_values[1:]:
            assert not torch.equal(X_first, X), "All Xs are expected to be different with each other..."
        ys_values = list(ys.values())
        y_first = ys_values[0]
        for y in ys_values[1:]:
            assert torch.equal(y_first, y), "All ys are expected to be same..."
        
        self.Xs = Xs
        self.y = y_first

        if verbose:
            print("\nTrials")
            for name, X in self.Xs.items():
                print(f"- shape of {name}", X.shape)
            print("- shape of y", self.y.shape)

    def __getitem__(self, cur_ind):
        inputs = {}
        for name, X in self.Xs.items():
            inputs[name] = X[cur_ind]
        labels = self.y[cur_ind]
        return inputs, labels

    def __len__(self):
        return len(self.y)


# In[2]:
# model.py

import torch
from torch import nn
from torch.nn import init
from torch.functional import F
from collections import OrderedDict
# from braindecode.models.eegnet import Conv2dWithConstraint


class Square(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x ** 2


# class Abs(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return torch.abs(x)


class Log(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(x)


# class Squeeze(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         x = torch.squeeze(x, dim=3)
#         return torch.squeeze(x, dim=2)


class TransposeTimeChannel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 1, 3, 2).contiguous()
    
    
class SelectBand(nn.Module):
    def __init__(self, input_band_name):
        super().__init__()
        self.input_band_name = input_band_name
        
    def __repr__(self):
        return "SelectBand(input_band_name={})".format(self.input_band_name)
    
    def forward(self, xs):
        """
        xs(dict) : input data
        """
        return xs[self.input_band_name]


# class SelectLocalRegion5x5(nn.Module):
#     local_regions_5x5 = {
#         1:[0, 1,2,3,4,5,7,8,9,10,11],
#         2:[1, 0, 2,3,6,7,8,9,13,14,15],
#         3:[2, 0,1, 3,4,6,7,8,9,10,13,14,15,16],
#         4:[3, 0,1,2, 4,5,7,8,9,10,11,13,14,15,16,17],
#         5:[4, 0,2,3, 5,8,9,10,11,12,14,15,16,17],
#         6:[5, 0,3,4, 9,10,11,12,15,16,17],
#         7:[6, 1,2, 7,8,13,14,18],
#         8:[7, 0,1,2,3,6, 8,9,13,14,15,18,19],
#         9:[8, 0,1,2,3,4,6,7, 9,10,13,14,15,16,18,19,20],
#         10:[9, 0,1,2,3,4,5,7,8, 10,11,13,14,15,16,17,18,19,20],
#         11:[10, 0,2,3,4,5,8,9, 11,12,14,15,16,17,18,19,20],
#         12:[11, 0,3,4,5,9,10, 12,15,16,17,19,20],
#         13:[12, 4,5,10,11, 16,17,20],
#         14:[13, 1,2,3,6,7,8,9, 14,15,18,19,21],
#         15:[14, 1,2,3,4,6,7,8,9,10,13, 15,16,18,19,20,21],
#         16:[15, 1,2,3,4,5,7,8,9,10,11,13,14 ,16,17,18,19,20,21],
#         17:[16, 2,3,4,5,8,9,10,11,12,14,15, 17,18,19,20,21],
#         18:[17, 3,4,5,9,10,11,12,15,16, 19,20,21],
#         19:[18, 6,7,8,9,10,13,14,15,16, 19,20,21],
#         20:[19, 7,8,9,10,11,13,14,15,16,17,18, 20,21],
#         21:[20, 8,9,10,11,12,14,15,16,17,18,19, 21],
#         22:[21, 13,14,15,16,17,18,19,20, ],
#         "all":list(range(22)),
#     } # from exp270
    
#     def __init__(self, local_region_id):
#         super().__init__()
#         self.local_region_id = local_region_id
        
#     def __repr__(self):
#         return "SelectLocalRegion5x5(local_region_id={})".format(self.local_region_id)
        
#     def forward(self, x):
#         """
#         x (torch.Tensor) : shape of (batch, 1, n_channels, n_time_smaples)
#         """
#         assert x.shape[2] == 22
#         return x[:, :, self.local_regions_5x5[self.local_region_id], :] # local region
        

class SelectLocalRegionHGD(nn.Module):
    local_regions_HGD = {
        1:[0, 1,2,6,7,8,13,14,15],
        2:[1, 0, 2,6,7,8,9,13,14,15],
        3:[2, 0,1, 3,6,7,8,9,10,13,14,15,16],
        4:[3, 2, 4,5,8,9,10,11,12,15,16,17,18],
        5:[4, 3, 5,9,10,11,12,16,17,18],
        6:[5, 3,4, 10,11,12,16,17,18],
        7:[6, 0,1,2, 7,8,13,14,15,19,20,21],
        8:[7, 0,1,2,6, 8,9,13,14,15,19,20,21],
        9:[8, 0,1,2,3,6,7, 9,10,13,14,15,16,19,20,21,22],
        10:[9, 1,2,3,4,7,8, 10,11,14,15,16,17,20,21,22,23],
        11:[10, 2,3,4,5,8,9, 11,12,15,16,17,18,21,22,23,24],
        12:[11, 3,4,5,9,10, 12,16,17,18,22,23,24],
        13:[12, 3,4,5,10,11, 16,17,18,22,23,24],
        14:[13, 0,1,2,6,7,8, 14,15,19,20,21,25,26,27],
        15:[14, 0,1,2,6,7,8,9,13, 15,19,20,21,25,26,27],
        16:[15, 0,1,2,3,6,7,8,9,10,13,14, 16,19,20,21,22,25,26,27,28],
        17:[16, 2,3,4,5,8,9,10,11,12,15, 17,18,21,22,23,24,27,28,29,30],
        18:[17, 3,4,5,9,10,11,12,16, 18,22,23,24,28,29,30],
        19:[18, 3,4,5,10,11,12,16,17, 22,23,24,28,29,30],
        20:[19, 6,7,8,13,14,15, 20,21,25,26,27,31,32,33],
        21:[20, 6,7,8,9,13,14,15,19, 21,25,26,27,31,32,33,34],
        22:[21, 6,7,8,9,10,13,14,15,16,19,20, 22,25,26,27,28,31,32,33,34,35],
        23:[22, 8,9,10,11,12,15,16,17,18,21, 23,24,27,28,29,30,33,34,35,36,37],
        24:[23, 9,10,11,12,16,17,18,22, 24,28,29,30,34,35,36,37],
        25:[24, 10,11,12,16,17,18,22,23, 28,29,30,35,36,37],
        26:[25, 13,14,15,19,20,21, 26,27,31,32,33,38,39,40],
        27:[26, 13,14,15,19,20,21,25, 27,31,32,33,34,38,39,40],
        28:[27, 13,14,15,16,19,20,21,22,25,26, 28,31,32,33,34,35,38,39,40,41],
        29:[28, 15,16,17,18,21,22,23,24,27, 29,30,33,34,35,36,37,40,41,42,43],
        30:[29, 16,17,18,22,23,24,28, 30,34,35,36,37,41,42,43],
        31:[30, 16,17,18,22,23,24,28,29, 35,36,37,41,42,43],
        32:[31, 19,20,21,25,26,27, 32,33,38,39,40],
        33:[32, 19,20,21,25,26,27,31, 33,34,38,39,40],
        34:[33, 19,20,21,22,25,26,27,28,31,32, 34,35,38,39,40,41],
        35:[34, 20,21,22,23,26,27,28,29,32,33, 35,36,39,40,41,42],
        36:[35, 21,22,23,24,27,28,29,30,33,34, 36,37,40,41,42,43],
        37:[36, 22,23,24,28,29,30,34,35, 37,41,42,43],
        38:[37, 22,23,24,28,29,30,35,36, 41,42,43],
        39:[38, 25,26,27,31,32,33, 39,40],
        40:[39, 25,26,27,31,32,33,34,38, 40],
        41:[40, 25,26,27,28,31,32,33,34,35,38,39, 41],
        42:[41, 27,28,29,30,33,34,35,36,37,40, 42,43],
        43:[42, 28,29,30,34,35,36,37,41, 43],
        44:[43, 28,29,30,35,36,37,41,42, ],
        "all":list(range(44)),
    } 
    
    def __init__(self, local_region_id):
        super().__init__()
        self.local_region_id = local_region_id
        
    def __repr__(self):
        return "SelectLocalRegionHGD(local_region_id={})".format(self.local_region_id)
        
    def forward(self, x):
        """
        x (torch.Tensor) : shape of (batch, 1, n_channels, n_time_smaples)
        """
        assert x.shape[2] == 44
        return x[:, :, self.local_regions_HGD[self.local_region_id], :] # local region
        
        
        
        
# class LazyLinearWithConstraint(nn.LazyLinear):
#     "reference: https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegnet.py"
#     def __init__(self, *args, max_norm=1, **kwargs):
#         self.max_norm = max_norm
#         nn.LazyLinear.__init__(self, *args, **kwargs)

#     def forward(self, x):
#         self.weight.data = torch.renorm(
#             self.weight.data, p=2, dim=0, maxnorm=self.max_norm
#         )
#         return nn.LazyLinear.forward(self, x)
    

def generate_subcnn_exp453(input_band_name, 
                            local_region_id,
                            kernel_size, 
                            padding=0,
                            temp_hidden=40,
                            spat_hidden=40,
                            pool_size=75,
                            pool_stride=15,
                            p_drop=0.5,
                            use_xavier_initialization=True, 
                            verbose=True):

    n_local_region_channels = len(SelectLocalRegionHGD.local_regions_HGD[local_region_id])
    
    model = nn.Sequential(OrderedDict([
        ("band", SelectBand(input_band_name=input_band_name)),
        ("local_region", SelectLocalRegionHGD(local_region_id=local_region_id)),
        ("transpose_time_channel", TransposeTimeChannel()),
        ("temp_conv", nn.Conv2d(1,  temp_hidden, 
                                              kernel_size=(kernel_size,1), 
                                              stride=(1,1), padding=padding)),
        ("spat_conv", nn.Conv2d(temp_hidden, spat_hidden, 
                                              kernel_size=(1,n_local_region_channels), 
                                              stride=(1,1), bias=False)),
        ("bn", nn.BatchNorm2d(spat_hidden)),
        ("square", Square()),
        ("mean_pool", nn.AvgPool2d(kernel_size=(pool_size,1), stride=(pool_stride,1))),
        ("log", Log()),
        ("dropout", nn.Dropout(p_drop)), # nn.Dropout(0.5) is used in braindecode...
        # ("clf_conv", nn.Conv2d(40,4, kernel_size=(69,1), stride=(1,1))),
        ("flatten", nn.Flatten()),
        ("clf", nn.LazyLinear(4)),
        # ("squeeze", Squeeze())
    ]))
    
    if use_xavier_initialization is True:
        # xavier initialization is used braindecode...
        # Initialization, xavier is same as in paper...
        init.xavier_uniform_(model.temp_conv.weight, gain=1)
        init.constant_(model.temp_conv.bias, 0)

        init.xavier_uniform_(model.spat_conv.weight, gain=1)

        init.constant_(model.bn.weight, 1)
        init.constant_(model.bn.bias, 0)

        # init.xavier_uniform_(model.clf_conv.weight, gain=1)
        # init.constant_(model.clf_conv.bias, 0)

    # To initialize lazy layer
    dummy_input = torch.zeros(size=(1,1,1125,n_local_region_channels))
    model[3:](dummy_input)

    if verbose is True:
        print(model)
        
    return model

class ParallelSubCNNs(nn.Module):
    def __init__(self, SubCNN, sub_cnn_dicts):
        super().__init__()
        sub_cnns = []
        for sub_cnn_name, sub_cnn_dict in sub_cnn_dicts.items():
            sub_cnns.append(
                (sub_cnn_name, SubCNN(**sub_cnn_dict))
            )
        self.sub_cnns = nn.ModuleDict(OrderedDict(sub_cnns))

    def forward(self, X):
        sub_outputs = OrderedDict()
        for sub_cnn_name, sub_cnn in self.sub_cnns.items():
            sub_output = sub_cnn(X)
            sub_outputs[sub_cnn_name] = sub_output
        return sub_outputs

    
class WeightCombiner_for_dict(nn.Module):
    def __init__(self, S):
        super().__init__()
        self.S = S
        self.weight_coeff = nn.Parameter(torch.ones(S) / S)
        
    def __repr__(self):
        return "WeightCombiner_for_dict(S={})".format(self.S)
        
    def forward(self, **sub_outputs):
        assert len(sub_outputs) == self.S 
        for sub_cnn_name, sub_output in sub_outputs.items():
            assert sub_output.ndim == 2
            assert sub_output.size(1) == 4 # (batch, n_classes)
        stacked = torch.stack(tuple(sub_outputs.values()), dim=1) # (batch, S, n_classes)
        weighted = stacked * self.weight_coeff.view(1,self.S,1) # broadcasting (batch, S, n_classes)
        return weighted.sum(dim=1)
        

class MultiCNN_exp453(nn.Module):
    def __init__(self, parallel_sub_cnns, weight_combiner):
        super().__init__()
        self.parallel_sub_cnns = parallel_sub_cnns
        self.weight_combiner = weight_combiner
        
    def forward(self, Xs):
        sub_outputs = self.parallel_sub_cnns(Xs)
        return self.weight_combiner(**sub_outputs)
    
    

# In[8]:
# utils.py

import torch
from torch.functional import F

def evaluate_trialwise_with_dataloader_without_ind(model, trial_dataloaders, mode, regularizer=None, prefix=""):
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
    total_labels = []
    with torch.no_grad():
        for inputs, labels in trial_dataloaders[mode]:
            total_outputs.append(model(inputs))
            total_labels.append(labels)

        total_outputs = torch.cat(total_outputs)
        total_labels = torch.cat(total_labels)
        total_preds = torch.softmax(total_outputs, dim=1).argmax(dim=1, keepdim=True)

        assert total_preds.shape == total_labels.shape
        total_corrects = total_preds == total_labels
        acc = torch.mean(total_corrects.float())
        ce_loss = F.cross_entropy(total_outputs, total_labels.flatten())

    if regularizer:
        return {
            f"{prefix}{mode}_acc": acc.item(),
            f"{prefix}{mode}_ce_loss": ce_loss.item(),
            f"{prefix}{mode}_reg_loss": regularizer(model).item()
        }
        
    else:
        return {
            f"{prefix}{mode}_acc": acc.item(),
            f"{prefix}{mode}_ce_loss": ce_loss.item(),
        }
        

def maxnorm_453(model):
    last_weight = None
    assert model.__class__ == nn.Sequential
    for name, module in list(model.named_modules()):
        if hasattr(module, "weight") and (
            "BatchNorm" not in module.__class__.__name__
        ):
            module.weight.data = torch.renorm(module.weight.data, 2, 0, maxnorm=2)
            last_weight = module.weight
    if last_weight is not None:
        last_weight.data = torch.renorm(last_weight.data, 2, 0, maxnorm=0.5)


def multi_maxnorm_exp453(multi_model):
    for name, model in list(multi_model.parallel_sub_cnns.module.sub_cnns.items()):
        maxnorm_453(model)
        
        
# In[ ]:
# main.py

import os
from torch.utils.data import DataLoader
import pandas as pd

def exp(rank, world_size, args):
    assert os.path.exists(args.result_dir)

    print("args:\n", args)
    subject = args.subject

    # DistributedDataParallel setup
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # 1. Data load  & 2. Preprocessing
    if (os.path.exists(args.preprocessed_data_path)
        ) and (args.use_preprocessed_data is True):
        # load preprocessed data
        data_path = args.preprocessed_data_path
        data = torch.load(data_path, map_location=torch.device(rank))
        Xs_tr = data["Xs_tr"]
        ys_tr = data["ys_tr"]
        Xs_te = data["Xs_te"]
        ys_te = data["ys_te"]
    elif (not os.path.exists(args.preprocessed_data_path)
        ) or (args.use_preprocessed_data is False):
        Xs_tr, ys_tr = load_and_preprocessing_for_input_band_dicts_exp442(args=args, train=True)
        Xs_te, ys_te = load_and_preprocessing_for_input_band_dicts_exp442(args=args, train=False)
    
        # 3. to Tensor
        print("\nTrain")
        Xs_tr, ys_tr = to_tensor(Xs_tr, ys_tr, device=rank)
        print("\nTest")
        Xs_te, ys_te = to_tensor(Xs_te, ys_te, device=rank)
    
        if not os.path.exists(args.preprocessed_data_path):
            data = OrderedDict()
            data["Xs_tr"] = Xs_tr
            data["ys_tr"] = ys_tr
            data["Xs_te"] = Xs_te
            data["ys_te"] = ys_te
            torch.save(data, args.preprocessed_data_path)
    else :
        raise
    
    
    # 4. Dataset 
    print("\nDataset")
    print("\ntrain")
    multi_dict_dataset_tr = MultiDictDataset(
        Xs=Xs_tr,
        ys=ys_tr,
        verbose=True
    )
    print("\ntrain")
    multi_dict_dataset_te = MultiDictDataset(
        Xs=Xs_te,
        ys=ys_te,
        verbose=True
        )

    # 5. DataLoader
    train_sampler = torch.utils.data.distributed.DistributedSampler(multi_dict_dataset_tr)
    multi_dict_dataloaders = {
        "tr": DataLoader(multi_dict_dataset_tr, batch_size=args.batch_size, 
                         shuffle=(train_sampler is None), sampler=train_sampler),
        "te": DataLoader(multi_dict_dataset_te, batch_size=args.batch_size, 
                         shuffle=False),
    } # for evaluation


    ## For Loop ##
    results = []
    for i_try in range(args.i_try_start, args.i_try_start + args.repeat):
        print("\n# TRY", i_try, "\n")
        
        
        # 6. Model
        # parallel sub cnns
        _parallel_sub_cnns = ParallelSubCNNs(
            SubCNN=generate_subcnn_exp453, 
            sub_cnn_dicts=args.sub_cnn_dicts
        )
        _parallel_sub_cnns = _parallel_sub_cnns.to(rank)
        _parallel_sub_cnns = nn.SyncBatchNorm.convert_sync_batchnorm(_parallel_sub_cnns).to(rank) # convert_sync_batchnorm
        _parallel_sub_cnns = nn.parallel.DistributedDataParallel(_parallel_sub_cnns, device_ids=[rank])
        # weight combiner
        _weight_combiner = WeightCombiner_for_dict(
            S=len(_parallel_sub_cnns.module.sub_cnns)
        )
        _weight_combiner = _weight_combiner.to(rank)
        _weight_combiner = nn.parallel.DistributedDataParallel(_weight_combiner, device_ids=[rank])
        # multi cnn model
        model = MultiCNN_exp453(_parallel_sub_cnns, _weight_combiner)
        

        # 7. Learning strategy
        loss_function = lambda outputs, labels: F.cross_entropy(outputs, labels.flatten())
        optimizer = torch.optim.Adam(model.parameters())
        lr_scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

        # 8. Epochs
        print("\nTraining")
        epoch_df = pd.DataFrame()            
        for epoch in range(1, args.epoch+1):
            train_sampler.set_epoch(epoch)
            
            # Train
            model.train()
            for inputs, labels in multi_dict_dataloaders["tr"]:  
                sub_outputs = model.parallel_sub_cnns(inputs) # dictionary
                outputs = model.weight_combiner(**sub_outputs) # tensor
                
                tentative_losses = OrderedDict()
                for sub_cnn_name, sub_output in sub_outputs.items():
                    tentative_loss = loss_function(sub_output, labels)
                    tentative_losses[sub_cnn_name] = tentative_loss
                
                overall_loss = loss_function(outputs, labels)

                if args.use_amalgamated_loss is True:
                    loss = (args.end_to_end_weight*overall_loss) + (args.tentative_weight*sum(tentative_losses.values()))
                elif args.use_amalgamated_loss is False:
                    loss = overall_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                multi_maxnorm_exp453(model)
                maxnorm_weight = lambda x: x/torch.norm(x, p=1) if torch.norm(x, p=1) > 1 else x
                model.weight_combiner.module.weight_coeff.data = maxnorm_weight(model.weight_combiner.module.weight_coeff.data)
            
            # step scheduler for every epoch
            lr_scheduler.step()
            
            # Evaluation
            if rank == 0: 
                # It is not DDP model, just local model.
                # To avoid dist.barrier hanging.
                model_module = MultiCNN_exp453(
                    parallel_sub_cnns=model.parallel_sub_cnns.module, 
                    weight_combiner=model.weight_combiner.module
                )
                evaluation_tr = evaluate_trialwise_with_dataloader_without_ind(
                    model_module,
                    trial_dataloaders=multi_dict_dataloaders,
                    mode="tr",
                    prefix="",
                )
                evaluation_te = evaluate_trialwise_with_dataloader_without_ind(
                    model_module,
                    trial_dataloaders=multi_dict_dataloaders,
                    mode="te",
                    prefix="",
                )
                # evaluation subs
                evaluation_subs = {}
                for sub_cnn_name, sub_cnn in model_module.parallel_sub_cnns.sub_cnns.items():
                    evaluation_sub_i_tr = evaluate_trialwise_with_dataloader_without_ind(
                        sub_cnn,
                        trial_dataloaders=multi_dict_dataloaders,
                        mode="tr",
                        prefix="{}_".format(sub_cnn_name),
                    )
                    evaluation_sub_i_te = evaluate_trialwise_with_dataloader_without_ind(
                        sub_cnn,
                        trial_dataloaders=multi_dict_dataloaders,
                        mode="te",
                        prefix="{}_".format(sub_cnn_name),
                    )
                    evaluation_subs.update(evaluation_sub_i_tr)
                    evaluation_subs.update(evaluation_sub_i_te)
                # update epoch_df
                assert len(epoch_df) == epoch-1
                epoch_df = epoch_df.append(
                    dict(
                        **evaluation_subs, 
                        **evaluation_tr, 
                        **evaluation_te,
                    ),
                    ignore_index=True,
                )
                print(datetime.datetime.now())
                print(args.result_dir, args.save_name)
                print("try", i_try, "subject", subject, "epoch", epoch)
                print(epoch_df.iloc[-1])
                print()
            
            dist.barrier()
            
        # After training
        if rank == 0:
            print("\nLast Epoch")
            print(epoch_df.iloc[-1])

            # result_name = f"{args.result_dir}/try{i_try}_subject{subject}_{args.save_name}"
            result_name = args.result_name.format(i_try)
            epoch_df.to_csv(result_name + ".csv")
            # torch.save(model.state_dict(), result_name + ".h5")
            torch.save(model_module.state_dict(), result_name + ".h5")

            # no more needed..
            results.append(round(epoch_df["te_acc"].iloc[-1], 2))

    # return results
    return 

class Args:
    # multi band
    input_band_dicts = OrderedDict(
        f_0_4 = OrderedDict(lowcut = 0, highcut = 4),
        f_2_10 = OrderedDict(lowcut = 2, highcut = 10),
        f_6_22 = OrderedDict(lowcut = 6, highcut = 22),
        f_16_high = OrderedDict(lowcut = 16, highcut = 0),
    )
    # band-dependent kernel size
    kernel_dicts = OrderedDict(
                        f_0_4  = 100,
                        f_2_10 = 50,
                        f_6_22 = 25,
                        f_16_high = 25,
    )
    assert kernel_dicts.keys() == input_band_dicts.keys()
    # sub-CNN configuration
    sub_cnn_dicts = OrderedDict()
    # for _local_region_id in ["all"]:
    for _local_region_id in range(1,45): # 44 channels
        for _input_band_name in input_band_dicts.keys():
            _sub_cnn_name = "sub_cnn_LR{}_{}".format(_local_region_id, _input_band_name)
            sub_cnn_dicts[_sub_cnn_name] = OrderedDict(
                input_band_name = _input_band_name,
                local_region_id = _local_region_id,
                kernel_size = kernel_dicts[_input_band_name],
                padding = 0,
                temp_hidden = 40,
                spat_hidden = 40,
                pool_size = 75,
                pool_stride = 15,
                p_drop = 0.5,
                use_xavier_initialization = True, 
                verbose = False
            )            
    # preprocessing
    fs_new = 250
    start_sec_offset = -0.5 # cue-0.5 ~ cue+4.0 sec
    stop_sec_offset = 4.0   # HGD has onset on cue.
    order = 3
    factor_new = 1e-3
    init_block_size = 1000
    # setting
    use_amalgamated_loss = True
    epoch = 500
    batch_size = 30 # 60; since 2 GPUs are used.
    lr_step_size = 300
    lr_gamma = 0.1
    # device = "cuda:2"
    CUDA_VISIBLE_DEVICES = "0, 1, 2"
    data_dir = "/home/jinhyo/JHS_server1/multi_class_motor_imagery/data/HGD"
    use_preprocessed_data = True
    preprocessed_data_path = None
    repeat = 1
    save_name = "HGD_higher_wider_overlap_4band_highpass_dependent3_kernel_cnn_local_region_alpha_maxnorm_L1_1_lossWeight_09_01"
    end_to_end_weight = 0.9 # for end-to-end cross entropy loss
    tentative_weight = 0.1 # for tentative cross entropy loss
    assert 1 == (end_to_end_weight+tentative_weight)
    
    def __init__(self, subject, i_try_start, result_dir):
        self.subject = subject
        self.i_try_start = i_try_start
        self.result_dir = result_dir
        self.preprocessed_data_path = "/home/jinhyo/JHS_server1/multi_class_motor_imagery/local_region_pruning/"
        # self.preprocessed_data_path += "cache/{}/{}_data_subject{}.h5".format(result_dir, result_dir, subject)  
        self.preprocessed_data_path += "cache/HGD_higher_wider_overlap_4band_highpass_lr_fix/f0-4_f2-10_f6-22_f16-high_subject{}.h5".format(subject)  
        self.result_name = f"{result_dir}/try{i_try_start}_subject{subject}_{self.save_name}"

    def __repr__(self):
        repr = ""
        repr = repr + "\n" + "\n".join([
            "{} = {}".format(attr, getattr(self, attr)) for attr in vars(self) 
            if (not attr.startswith("_")) and (attr !="sub_cnn_dicts")
        ])
        repr = repr + "\n" + "\n".join([
            "{} = {}".format(attr, getattr(Args, attr)) for attr in vars(Args) 
            if (not attr.startswith("_")) and (attr !="sub_cnn_dicts")
        ])
        return repr
    
# # In[ ]:
if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"]= Args.CUDA_VISIBLE_DEVICES  # Set the GPUs 2 and 3 to use
    world_size = torch.cuda.device_count()
    import time
    wait_for_file = "/home/jinhyo/JHS_server1/multi_class_motor_imagery/local_region_pruning/exp467/try10_subject9_higher_wider_overlap_4band_dependent3_kernel_cnn_alpha_L1norm_1_lossWeight_09_01_local_region.h5"
    result_dir = __file__.split("/")[-1].split(".")[0]
    while not os.path.exists(wait_for_file):
        print("\n\n\n\n")
        print("*" * 60)
        print("I am {}".format(result_dir))
        print("Not yet created:\n{}".format(wait_for_file))
        print("*" * 60)
        print("\n\n\n\n")
        time.sleep(60)
    
    for i_try_start in range(1, 11):
        for subject in range(1,15):
            result_dir = __file__.split("/")[-1].split(".")[0]
            args = Args(subject=subject, i_try_start=i_try_start, result_dir=result_dir)
            assert args.repeat == 1
            if (os.path.exists(args.result_name+".csv")) and (os.path.exists(args.result_name+".h5")):
                print("\n\n\n\n")
                print("*" * 60)
                print("Already exist & Pass:\n{}".format(args.result_name))
                print("*" * 60)
                print("\n\n\n\n")
                continue
            mp.spawn(exp,
                     args=(world_size, args),
                     nprocs=world_size) 
                

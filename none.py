#!/usr/bin/env python
# coding: utf-8

# In[1]:
# data.py

from ast import Or
from unittest import result
import numpy as np
import torch
from collections import OrderedDict
import logging
import sys
sys.path += ["/home/jinhyo/JHS_server2/multi_class_motor_imagery"]
from bcitools.bcitools import (load_gdf2mat_feat_mne,
                               s_to_cnt,
                               rerange_label_from_0,
                               rerange_pos_from_0,
                               drop_eog_from_cnt,
                               replace_break_with_mean,
                               change_scale,
                               butter_bandpass_filter,
                               exponential_moving_standardize_from_braindecode,
                               epoch_X_y_from_data)

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

def load_and_preprocessing_for_input_band_dicts_exp346(args, train=True):
    order = args.order
    factor_new = args.factor_new
    init_block_size = args.init_block_size
    input_band_dicts = args.input_band_dicts
    subject = args.subject
    data_dir = args.data_dir
    start_sec_offset = args.start_sec_offset
    stop_sec_offset = args.stop_sec_offset
    
    Xs, ys = OrderedDict(), OrderedDict()
    train_test = "Train" if train is True else "Test"
    for input_band_name, input_band_dict in input_band_dicts.items():
        print(f"\nLoad {train_test} Data for {input_band_name}")
        data_sub = load_gdf2mat_feat_mne(
            subject=subject,
            train=train,
            data_dir=data_dir,
            overflowdetection=False,
            verbose=True,
        )
        data_sub = s_to_cnt(data_sub, verbose=True)
        data_sub = rerange_label_from_0(data_sub, verbose=True)
        data_sub = rerange_pos_from_0(data_sub, verbose=True)

        print(f"\nPreprocessing for {input_band_name}")
        data_sub = drop_eog_from_cnt(data_sub, verbose=True)
        data_sub = replace_break_with_mean(data_sub, verbose=True)
        data_sub = change_scale(data_sub, factor=1e06, channels="all", verbose=True)
        data_sub = butter_bandpass_filter(
            data_sub, lowcut=input_band_dict["lowcut"], highcut=input_band_dict["highcut"], order=order, verbose=True
        )
        data_sub = exponential_moving_standardize_from_braindecode(
            data_sub, factor_new=factor_new, init_block_size=init_block_size, verbose=True
        )
        X_sub, y_sub = epoch_X_y_from_data(
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


class SelectLocalRegion5x5(nn.Module):
    local_regions_5x5 = {
        1:[0, 1,2,3,4,5,7,8,9,10,11],
        2:[1, 0, 2,3,6,7,8,9,13,14,15],
        3:[2, 0,1, 3,4,6,7,8,9,10,13,14,15,16],
        4:[3, 0,1,2, 4,5,7,8,9,10,11,13,14,15,16,17],
        5:[4, 0,2,3, 5,8,9,10,11,12,14,15,16,17],
        6:[5, 0,3,4, 9,10,11,12,15,16,17],
        7:[6, 1,2, 7,8,13,14,18],
        8:[7, 0,1,2,3,6, 8,9,13,14,15,18,19],
        9:[8, 0,1,2,3,4,6,7, 9,10,13,14,15,16,18,19,20],
        10:[9, 0,1,2,3,4,5,7,8, 10,11,13,14,15,16,17,18,19,20],
        11:[10, 0,2,3,4,5,8,9, 11,12,14,15,16,17,18,19,20],
        12:[11, 0,3,4,5,9,10, 12,15,16,17,19,20],
        13:[12, 4,5,10,11, 16,17,20],
        14:[13, 1,2,3,6,7,8,9, 14,15,18,19,21],
        15:[14, 1,2,3,4,6,7,8,9,10,13, 15,16,18,19,20,21],
        16:[15, 1,2,3,4,5,7,8,9,10,11,13,14 ,16,17,18,19,20,21],
        17:[16, 2,3,4,5,8,9,10,11,12,14,15, 17,18,19,20,21],
        18:[17, 3,4,5,9,10,11,12,15,16, 19,20,21],
        19:[18, 6,7,8,9,10,13,14,15,16, 19,20,21],
        20:[19, 7,8,9,10,11,13,14,15,16,17,18, 20,21],
        21:[20, 8,9,10,11,12,14,15,16,17,18,19, 21],
        22:[21, 13,14,15,16,17,18,19,20, ],
        "all":list(range(22)),
    } # from exp270
    
    def __init__(self, local_region_id):
        super().__init__()
        self.local_region_id = local_region_id
        
    def __repr__(self):
        return "SelectLocalRegion5x5(local_region_id={})".format(self.local_region_id)
        
    def forward(self, x):
        """
        x (torch.Tensor) : shape of (batch, 1, n_channels, n_time_smaples)
        """
        assert x.shape[2] == 22
        return x[:, :, self.local_regions_5x5[self.local_region_id], :] # local region
        
        
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
    

def generate_subcnn_exp346(input_band_name, 
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

    n_local_region_channels = len(SelectLocalRegion5x5.local_regions_5x5[local_region_id])
    
    model = nn.Sequential(OrderedDict([
        ("band", SelectBand(input_band_name=input_band_name)),
        ("local_region", SelectLocalRegion5x5(local_region_id=local_region_id)),
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
    
    if verbose is True:
        print(model)
        
    return model

    
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
        

class MultiCNN_exp346(nn.Module):
    def __init__(self, SubCNN, sub_cnn_dicts):
        super().__init__()
        sub_cnns = []
        for sub_cnn_name, sub_cnn_dict in sub_cnn_dicts.items():
            sub_cnns.append(
                (sub_cnn_name, SubCNN(**sub_cnn_dict))
            )
        self.sub_cnns = nn.ModuleDict(OrderedDict(sub_cnns))
        self.weight_combiner = WeightCombiner_for_dict(S=len(self.sub_cnns))
        
    def forward(self, Xs):
        sub_outputs = self.tentative_forward(Xs)
        return self.weight_combiner(**sub_outputs)
    
    def tentative_forward(self, X):
        sub_outputs = OrderedDict()
        for sub_cnn_name, sub_cnn in self.sub_cnns.items():
            sub_output = sub_cnn(X)
            sub_outputs[sub_cnn_name] = sub_output
        return sub_outputs
    

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
        

def maxnorm(model):
    last_weight = None
    assert model.__class__ == nn.Sequential
    for name, module in list(model.named_modules()):
        if hasattr(module, "weight") and (
            not module.__class__.__name__.startswith("BatchNorm")
        ):
            module.weight.data = torch.renorm(module.weight.data, 2, 0, maxnorm=2)
            last_weight = module.weight
    if last_weight is not None:
        last_weight.data = torch.renorm(last_weight.data, 2, 0, maxnorm=0.5)


def multi_maxnorm(multi_model):
    for name, model in list(multi_model.sub_cnns.items()):
        maxnorm(model)
        
        
# In[ ]:
# main.py

import os
from torch.utils.data import DataLoader
import pandas as pd

def exp(args):
    assert os.path.exists(args.result_dir)

    print("args:\n", args)
    subject = args.subject
    device = args.device


    # 1. Data load  & 2. Preprocessing
    total_Xs_tr = OrderedDict()
    total_ys_tr = OrderedDict()
    total_Xs_te = OrderedDict()
    total_ys_te = OrderedDict()
    for s in range(1,10):
        # load subject data
        if args.use_preprocessed_data is True:
            # load preprocessed data
            data_path = args.preprocessed_data_path_format.format(s)
            data = torch.load(data_path, map_location=args.device)
            total_Xs_tr[s] = data["Xs_tr"]
            total_ys_tr[s] = data["ys_tr"]
            total_Xs_te[s] = data["Xs_te"]
            total_ys_te[s] = data["ys_te"]
        else :
            raise
    
    # target subject
    _ = total_Xs_tr.pop(subject)
    assert len(total_Xs_tr) == 8
    _ = total_ys_tr.pop(subject)
    assert len(total_ys_tr) == 8
    Xs_te = total_Xs_te.pop(subject)
    ys_te = total_ys_te.pop(subject)
    # nontarget subjects
    Xs_tr = OrderedDict()
    ys_tr = OrderedDict()
    for key in Xs_te.keys():
        temp_X = []
        temp_y = []
        for s in total_Xs_tr.keys():
            temp_X.append(total_Xs_tr[s][key])
            temp_X.append(total_Xs_te[s][key])
            temp_y.append(total_ys_tr[s][key])
            temp_y.append(total_ys_te[s][key])
        Xs_tr[key] = torch.cat(temp_X)
        ys_tr[key] = torch.cat(temp_y)
            
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
    multi_dict_dataloaders = {
        "tr": DataLoader(multi_dict_dataset_tr, batch_size=args.batch_size, shuffle=True),
        "te": DataLoader(multi_dict_dataset_te, batch_size=args.batch_size, shuffle=False),
    } # for evaluation


    ## For Loop ##
    results = []
    for i_try in range(args.i_try_start, args.i_try_start + args.repeat):
        print("\n# TRY", i_try, "\n")
        
        
        # 6. Model
        model = MultiCNN_exp346(SubCNN=generate_subcnn_exp346, 
                                sub_cnn_dicts=args.sub_cnn_dicts)
        model = model.to(device)
        # # To initialize Lazy layer.
        # for dummy_input, dummy_label, _ in first_training_dataloader:
        #     print("shape of input", dummy_input.shape)
        #     dummy_output = model(dummy_input)
        #     print("shape of output", dummy_output.shape)
        #     break
        

        # 7. Learning strategy
        loss_function = lambda outputs, labels: F.cross_entropy(outputs, labels.flatten())
        optimizer = torch.optim.Adam(model.parameters())
        lr_scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

        # 8. Epochs
        print("\nTraining")
        epoch_df = pd.DataFrame()            
        for epoch in range(1, args.epoch+1):
            # Train
            model.train()
            for inputs, labels in multi_dict_dataloaders["tr"]:  
                sub_outputs = model.tentative_forward(inputs) # dictionary
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
                
                multi_maxnorm(model)
                maxnorm_weight = lambda x: x/torch.norm(x, p=1) if torch.norm(x, p=1) > 1 else x
                model.weight_combiner.weight_coeff.data = maxnorm_weight(model.weight_combiner.weight_coeff.data)
                
            # Evaluation
            evaluation_tr = evaluate_trialwise_with_dataloader_without_ind(
                model,
                trial_dataloaders=multi_dict_dataloaders,
                mode="tr",
                prefix="",
            )
            evaluation_te = evaluate_trialwise_with_dataloader_without_ind(
                model,
                trial_dataloaders=multi_dict_dataloaders,
                mode="te",
                prefix="",
            )
            # evaluation subs
            evaluation_subs = {}
            for sub_cnn_name, sub_cnn in model.sub_cnns.items():
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
            print(args.result_dir, args.save_name)
            print("try", i_try, "subject", subject, "epoch", epoch)
            print(epoch_df.iloc[-1])
            print()
            
            # step scheduler for every epoch
            lr_scheduler.step()
            
        # After training
        print("\nLast Epoch")
        print(epoch_df.iloc[-1])

        # result_name = f"{args.result_dir}/try{i_try}_subject{subject}_{args.save_name}"
        result_name = args.result_name.format(i_try)
        epoch_df.to_csv(result_name + ".csv")
        torch.save(model.state_dict(), result_name + ".h5")

        # 
        results.append(round(epoch_df["te_acc"].iloc[-1], 2))

    return results

class Args:
    # multi band
    input_band_dicts = OrderedDict(
        f_0_4 = OrderedDict(lowcut = 0, highcut = 4),
        f_2_10 = OrderedDict(lowcut = 2, highcut = 10),
        f_6_22 = OrderedDict(lowcut = 6, highcut = 22),
        f_16_38 = OrderedDict(lowcut = 16, highcut = 38),
    )
    # band-dependent kernel size
    kernel_dicts = OrderedDict(
                        f_0_4  = 100,
                        f_2_10 = 50,
                        f_6_22 = 25,
                        f_16_38 = 25,
    )
    assert kernel_dicts.keys() == input_band_dicts.keys()
    # sub-CNN configuration
    sub_cnn_dicts = OrderedDict()
    # for _local_region_id in range(1,23):
    for _local_region_id in ["all"]:
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
    start_sec_offset = 1.5 # cue-0.5 ~ cue+4.0 sec
    stop_sec_offset = 6.0
    order = 3
    factor_new = 1e-3
    init_block_size = 1000
    # setting
    use_amalgamated_loss = True
    epoch = 500
    batch_size = 60
    lr_step_size = 300
    lr_gamma = 0.1
    device = "cuda:1"
    data_dir = "/home/jinhyo/JHS_server2/multi_class_motor_imagery/data/BCICIV_2a/gdf"
    use_preprocessed_data = True
    preprocessed_data_path = None
    repeat = 1
    save_name = "subject_independent_mbkcnn"
    end_to_end_weight = 0.9 # for end-to-end cross entropy loss
    tentative_weight = 0.1 # for tentative cross entropy loss
    assert 1 == (end_to_end_weight+tentative_weight)
    
    def __init__(self, subject, i_try_start, result_dir):
        self.subject = subject
        self.i_try_start = i_try_start
        self.result_dir = result_dir
        self.preprocessed_data_path_format = "/home/jinhyo/JHS_server2/multi_class_motor_imagery/local_region_pruning/"
        self.preprocessed_data_path_format += "cache/higher_wider_overlap_4band/f0-4_f2-10_f6-22_f16-38_subject{}.h5"  
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
    for i_try_start in range(1, 2):
        for subject in range(1,10):
            result_dir = "/home/jinhyo/JHS_server2/multi_class_motor_imagery/SI_mbk-cnn/"
            result_dir += __file__.split("/")[-1].split(".")[0]
            args = Args(subject=subject, i_try_start=i_try_start, result_dir=result_dir)
            assert args.repeat == 1
            if (os.path.exists(args.result_name+".csv")) and (os.path.exists(args.result_name+".h5")):
                print("\n\n\n\n")
                print("*" * 60)
                print("Already exist & Pass:\n{}".format(args.result_name))
                print("*" * 60)
                print("\n\n\n\n")
                continue
            exp(args) 

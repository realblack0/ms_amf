from ms_amf.main_tools import make_results_directory, train_ms_amf
from ms_amf.data import load_gdf2mat, preprocess
from ms_amf.model import MultiCSP, SEBlock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.functional import F
import pandas as pd

import pickle
import argparse

from sklearn.model_selection import StratifiedKFold


###################
#### Modeling #####
###################

class Model(nn.Module):
    def __init__(
        self, 
        # MODEL HYPER PARAMETER ,
    ):
        super().__init__()

        # SPATIAL MULTI-SCALE MODULE 
        self.central_conv  = nn.Conv2d(1, 256, kernel_size=(16,1), stride=(16, 1), padding=0)
        self.parietal_conv = nn.Conv2d(1, 256, kernel_size=(16,1), stride=(16, 1), padding=0)
        
        # DENSE FUSION MODULE I
        # Temporal Multi-scale Module 1
        self.bn1 = nn.BatchNorm2d(512)
        self.temporal_conv1   = nn.Conv2d(512, 128, kernel_size=(1,25), stride=(1, 1), padding=(0,12))
        self.temporal_atrous1 = nn.Conv2d(512, 128, kernel_size=(1, 3), stride=(1, 1), padding=(0,2), dilation=2)
        self.am1 = SEBlock(c=128, r=4)
        self.am2 = SEBlock(c=128, r=4)
        
        # Temporal Multi-scale Module 2
        self.bn2 = nn.BatchNorm2d(768)
        self.temporal_conv2   = nn.Conv2d(768, 128, kernel_size=(1,25), stride=(1, 1), padding=(0,12))
        self.temporal_atrous2 = nn.Conv2d(768, 128, kernel_size=(1, 3), stride=(1, 1), padding=(0,2), dilation=2)
        self.am3 = SEBlock(c=128, r=4)
        self.am4 = SEBlock(c=128, r=4)
        
        # Temporal Multi-scale Module 3
        self.bn3 = nn.BatchNorm2d(1024)
        self.temporal_conv3   = nn.Conv2d(1024, 128, kernel_size=(1,25), stride=(1, 1), padding=(0,12))
        self.temporal_atrous3 = nn.Conv2d(1024, 128, kernel_size=(1, 3), stride=(1, 1), padding=(0,2), dilation=2)
        self.am5 = SEBlock(c=128, r=4)
        self.am6 = SEBlock(c=128, r=4)
        
        self.am7 = SEBlock(c=1280, r=4)
        self.ave_pool1 = nn.AvgPool2d(kernel_size=(1,25), stride=(1,25))
        
        
        # DENSE FUSION MODULE II
        # Temporal Multi-scale Module 1
        self.bn4 = nn.BatchNorm2d(1280)
        self.temporal_conv4   = nn.Conv2d(1280, 128, kernel_size=(1,3), stride=(1, 1), padding=(0,1))
        self.temporal_atrous4 = nn.Conv2d(1280, 128, kernel_size=(1,3), stride=(1, 1), padding=(0,2), dilation=2)
        self.am8 = SEBlock(c=128, r=4)
        self.am9 = SEBlock(c=128, r=4)
        
        # Temporal Multi-scale Module 2
        self.bn5 = nn.BatchNorm2d(1280)
        self.temporal_conv5   = nn.Conv2d(1536, 128, kernel_size=(1,3), stride=(1, 1), padding=(0,1))
        self.temporal_atrous4 = nn.Conv2d(1536, 128, kernel_size=(1,3), stride=(1, 1), padding=(0,2), dilation=2)
        self.am10 = SEBlock(c=128, r=4)
        self.am11 = SEBlock(c=128, r=4)
        
        # Temporal Multi-scale Module 3
        self.bn6 = nn.BatchNorm2d(1792)
        self.temporal_conv6   = nn.Conv2d(1792, 128, kernel_size=(1,3), stride=(1, 1), padding=(0,1))
        self.temporal_atrous6 = nn.Conv2d(1792, 128, kernel_size=(1,3), stride=(1, 1), padding=(0,2), dilation=2)
        self.am12 = SEBlock(c=128, r=4)
        self.am13 = SEBlock(c=128, r=4)
        
        self.am14 = SEBlock(c=2048, r=4)
        self.ave_pool2 = nn.AvgPool2d(kernel_size=(1,5), stride=(1,5))
        
        
        # OUTPUT
        self.bn7 = nn.BatchNorm2d(2048)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1*4*2048, 4)
        
    def forward(self, Zc, Zp): 
        """ 
        Args
        ----
            Zc (batch, channel, height, width) 
            Zp (batch, channel, height, width) 
        """
        #  H(·): BN − ReLU− [Conv (·) + AM (·)] &[Atrous Conv (·) + AM (·)] − Concat(·) − AM (·),
        
        # SPATIAL MULTI-SCALE MODULE 
        Fc = self.central_conv(Zc)
        Fp = self.parietal_conv(Zp)
        x0 = torch.cat([Fc, Fp], dim=1) # dim 512
        
        # DENSE FUSION MODULE I
        # Temporal Multi-scale Module 1
        x = self.bn1(x0)
        x = F.relu(x)
        
        t1 = self.temporal_conv1(x) 
        t1 = self.am1(t1) # dim 128
        
        t2 = self.temporal_atrous1(x)
        t2 = self.am2(t2) # dim 128
        
        x1 = torch.cat([t1,t2], dim=1) # dim 256
                
        # Temporal Multi-scale Module 2
        x = torch.cat([x0,x1], dim=1) # dim 768
        x = self.bn2(x)
        x = F.relu(x)
        
        t1 = self.temporal_conv2(x)
        t1 = self.am3(t1) # dim 128
        
        t2 = self.temporal_atrous2(x)
        t2 = self.am4(t2) # dim 128
        
        x2 = torch.cat([t1,t2], dim=1) # dim 256
                
        # Temporal Multi-scale Module 3
        x = torch.cat([x0,x1,x2], dim=1) # dim 1024
        x = self.bn3(x)
        x = F.relu(x)
        
        t1 = self.temporal_conv3(x)
        t1 = self.am5(t1) # dim 128
        
        t2 = self.temporal_atrous3(x)
        t2 = self.am6(t2) # dim 128
        
        x = torch.cat([x0,x1,x2,t1,t2], dim=1) # dim 1280
        x = self.am7(x)
        x3 = self.ave_pool1(x) 
        
        
        # DENSE FUSION MODULE II
        # Temporal Multi-scale Module 1
        x = self.bn4(x3)
        x = F.relu(x)
        
        t1 = self.temporal_conv4(x)  
        t1 = self.am8(t1) # dim 128
        
        t2 = self.temporal_atrous4(x)
        t2 = self.am9(t2) # dim 128
        
        x4 = torch.cat([t1,t2], dim=1) # dim 256
                
        # Temporal Multi-scale Module 2
        x = torch.cat([x3, x4], dim=1) # dim 1536
        x = self.bn5(x)
        x = F.relu(x)
        
        t1 = self.temporal_conv5(x)
        t1 = self.am10(t1) # dim 128
        
        t2 = self.temporal_atrous5(x)
        t2 = self.am11(t2) # dim 128
        
        x5 = torch.cat([t1,t2], dim=1)
                
        # Temporal Multi-scale Module 3
        x = torch.cat([x3, x4, x5], dim=1) # dim 1792
        x = self.bn6(x)
        x = F.relu(x)
        
        t1 = self.temporal_conv6(x)
        t1 = self.am12(t1) # dim 128
        
        t2 = self.temporal_atrous6(x)
        t2 = self.am13(t2) # dim 128
        
        x = torch.cat([x3,x4,x5,t1,t2], dim=1) # dim 2048
        x = self.am14(x)
        x6 = self.ave_pool2(x) 
        
        # OUTPUT
        x = self.bn7(x6)
        x = F.relu(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out
        
        
class SpatialEEGDataset(Dataset):
    def __init__(self, Zc, Zp, y):
        self.Zc = Zc
        self.Zp = Zp
        self.y = y

    def __getitem__(self, index):
        return self.Zc[index], self.Zp[index], self.y[index]

    def __len__(self):
        return len(self.y)

        
        
if __name__ == "__main__":
    ###################
    ## Configuration ##
    ###################
    # SYSTEM
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--seed', default="n")
    parser.add_argument('--data_dir', default="/home/jinhyo/multi_class_motor_imagery/data/BCICIV_2a/gdf")
    args = parser.parse_args()

    device = torch.device(args.device)
    name = args.name
    results_dir = make_results_directory(name, copy_file=__file__, copy_dir="ms_amf")

    # LEARNING STRATEGY
    batch_size = 20
    epochs     = 300
    criterion  = nn.CrossEntropyLoss()
    Optimizer  = torch.optim.Adam
    lr         = 10e-5
    print("batch_size", batch_size)
    print("epochs", epochs)
    print("criterion", criterion)
    print("Optimizer", Optimizer)
    print("lr", lr)

    ###################
    #### Load Data ####
    ###################
    data_tr = load_gdf2mat(subject           = args.subject, 
                           train             = True, 
                           data_dir          = args.data_dir,
                           overflowdetection = False)
    X_train, y_train = preprocess(s         = data_tr["s"], 
                            labels    = data_tr["labels"],
                            ch_names  = data_tr["ch_names"], 
                            pos       = data_tr["pos"], 
                            typ       = data_tr["typ"], 
                            fs        = data_tr["fs"], 
                            artifacts = data_tr["artifacts"],
                            filename  = data_tr["filename"], 
                            reject_trial = True)
    y_train = y_train - 1 # y_train : class label 1~4 -> 0~3
    
    data_te = load_gdf2mat(subject           = args.subject, 
                           train             = False, 
                           data_dir          = args.data_dir,
                           overflowdetection = False)
    X_test, y_test = preprocess(s         = data_te["s"], 
                            labels    = data_te["labels"],
                            ch_names  = data_te["ch_names"], 
                            pos       = data_te["pos"], 
                            typ       = data_te["typ"], 
                            fs        = data_te["fs"], 
                            artifacts = data_te["artifacts"],
                            filename  = data_te["filename"],
                            reject_trial = True)    
    y_test = y_test - 1 # y_test : class label 1~4 -> 0~3
       
    #########################
    # 10 Cross Validation   #
    #########################
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
        
    cv_hist = []
    skf = StratifiedKFold(n_splits=10, random_state=2020011075, shuffle=True)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"\n ###### fold {fold} ######")
        print("train_index", train_index)
        print("test_index", test_index)
        
        X_tr = X[train_index]
        y_tr = y[train_index]
        X_te = X[test_index]
        y_te = y[test_index]
        
        # Multi region
        print("\nMulti Region")
        C_tr = X_tr[:,:13,:] # batch, EEG channel, time
        P_tr = X_tr[:,13:,:]

        C_te = X_te[:,:13,:]
        P_te = X_te[:,13:,:]
        print("shape of C_tr", C_tr.shape)
        print("shape of P_tr", P_tr.shape)
        print("shape of C_te", C_te.shape)
        print("shape of P_te", P_te.shape)

        
        # CSP one-versus-rest
        print("\nCSP one-versus-rest")
        mcsp_c = MultiCSP()
        mcsp_c.fit(C_tr, y_tr)
        Zc_tr = mcsp_c.transform(C_tr) # (trials, EEG channels, time) == (B, H, W)
        Zc_te = mcsp_c.transform(C_te)

        mcsp_p = MultiCSP()
        mcsp_p.fit(P_tr, y_tr)
        Zp_tr = mcsp_p.transform(P_tr)
        Zp_te = mcsp_p.transform(P_te)
        print("shape of Zc_tr", Zc_tr.shape)
        print("shape of Zp_tr", Zp_tr.shape)
        print("shape of Zc_te", Zc_te.shape)
        print("shape of Zp_te", Zp_te.shape)

        
        # Tensor
        print("\nTensor")
        Zc_tr = torch.Tensor(np.expand_dims(Zc_tr,1)).to(device) # (trials, 1, EEG channels, time) == (B, C, H, W)
        Zp_tr = torch.Tensor(np.expand_dims(Zp_tr,1)).to(device)
        y_tr  = torch.tensor(y_tr.squeeze(), dtype=torch.long).to(device) 

        Zc_te = torch.Tensor(np.expand_dims(Zc_te,1)).to(device)
        Zp_te = torch.Tensor(np.expand_dims(Zp_te,1)).to(device)
        y_te  = torch.tensor(y_te.squeeze(), dtype=torch.long).to(device)
        print("shape of Zc_tr", Zc_tr.shape)
        print("shape of Zp_tr", Zp_tr.shape)
        print("shape of Zc_te", Zc_te.shape)
        print("shape of Zp_te", Zp_te.shape)


        ####################
        #    Training      #
        ####################

        train_dataset = SpatialEEGDataset(Zc_tr, Zp_tr, y_tr)
        test_dataset  = SpatialEEGDataset(Zc_te, Zp_te, y_te)

        if args.seed == "y":
            current_seed = 2021010556
            print("current_seed:", current_seed)
            torch.manual_seed(current_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        model = Model().to(device)
        
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        optimizer = Optimizer(model.parameters(), lr=lr)

        hist =  train_ms_amf(
                  name         = name, 
                  tag          = fold,
                  model        = model, 
                  # 
                  train_loader = train_loader, 
                  val_loader   = test_loader, 
                  epochs       = epochs,
                  device       = device,
                  # 
                  criterion     = criterion, 
                  optimizer     = optimizer,
                  #
                  results_dir   = results_dir,
        )

        # Save the result
        cv_hist.append(hist)    
        with open(f"{results_dir}/histories_{fold}.pkl", "wb") as f:
            pickle.dump(hist, f)
            
        print("\nBest Val Acc:", [_["val_acc"] for _ in hist["val_hist"]][hist["best_val_acc_epoch"]])
        print("\nBest Val Acc epoch:", hist["best_val_acc_epoch"])

    # 10 fold cross validation result (last epoch)
    print("\nLast epoch")
    running_acc = 0
    for i, hist in enumerate(cv_hist):
        acc = hist["val_hist"][-1]['val_acc']
        acc = round(acc, 2)
        running_acc += acc
        print(f"fold {i}: {acc:.2f}")
    print(f"mean: {running_acc/10:.2f}")
    
    # 10 fold cross validation result (best epoch)
    print("\nBest epoch")
    running_acc = 0
    for i, hist in enumerate(cv_hist):
        best_epoch = hist["best_val_acc_epoch"]
        acc = hist["val_hist"][best_epoch]['val_acc']
        acc = round(acc, 2)
        running_acc += acc
        print(f"fold {i}: {acc:.2f} ({best_epoch})")
    print(f"mean: {running_acc/10:.2f}")
    
    # 최종 결과 저장
    with open(f"{results_dir}/cv_hist.pkl", "wb") as f:
        pickle.dump(cv_hist, f)
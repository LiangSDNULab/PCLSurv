import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import math


class CanDataset(Dataset):
    def __init__(self, RNASeq_dataframe, miRNA_dataframe, clinical_dataframe, ind, device):

        self.RNASeq_feature = np.array(RNASeq_dataframe)
        self.miRNA_feature = np.array(miRNA_dataframe)
        self.clinical_feature = np.array(clinical_dataframe)
        self.ystatus = np.squeeze(self.clinical_feature[:, 1]).astype(float)
        self.ytime = np.squeeze(self.clinical_feature[:, 2]).astype(float)

        self.RNASeq_feature_data = self.RNASeq_feature[ind,]
        self.miRNA_feature_data = self.miRNA_feature[ind,]
        self.ystatus_data = self.ystatus[ind,]
        self.ytime_data = self.ytime[ind,]

        self.RNASeq_tensor = torch.tensor(self.RNASeq_feature_data, dtype=torch.float).to(device)
        self.miRNA_tensor = torch.tensor(self.miRNA_feature_data, dtype=torch.float).to(device)
        self.ystatus_tensor = torch.tensor(self.ystatus_data, dtype=torch.float).to(device)
        self.ytime_tensor = torch.tensor(self.ytime_data, dtype=torch.float).to(device)

    def __len__(self):
        return self.RNASeq_tensor.shape[0]

    def __getitem__(self, idx):
        RNASeq = self.RNASeq_tensor[idx]
        miRNA = self.miRNA_tensor[idx]
        ystatus = self.ystatus_tensor[idx]
        ytime = self.ytime_tensor[idx]

        return RNASeq, miRNA, ystatus, ytime




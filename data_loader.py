import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


class Dataset(Dataset):
    def __init__(self, filepath,  huayanfilepath,  transform):

        self.filepath = filepath
        self.transform = transform
        self.huayanfilepath = huayanfilepath
        self.labels = {'HL':0,'HCM': 1, 'DCM':2}
        self.file = os.listdir(self.filepath)
        self.huayanfile = pd.read_excel(huayanfilepath)
        self.data = []
        self.id = []
        rowNum = self.huayanfile.shape[0]
        for i in range(0,rowNum):
            x = self.huayanfile.iloc[i, 1:36]
            self.id.append(self.huayanfile.iloc[i, 36])
            self.data.append(x)
        self.data = MinMaxScaler().fit_transform(np.array(self.data))#归一化
        self.data = torch.tensor(self.data)
        self.data = self.data.float()

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        image = Image.open(self.filepath + os.sep + self.file[idx])
        return self.transform(image), self.data[self.id.index(int(self.file[idx].split('-')[1].split('.')[0].split('_')[0]))], self.labels[self.file[idx].split('-')[0]]
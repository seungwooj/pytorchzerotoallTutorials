# Exercise 8-1 - 같이 해봤으면 좋겠음.
# 1. Build DataLoader for Titanic data set.
# 2. Build a classifier using the DataLoader.

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TitanicDataSet(Dataset):

    def __init__(self):
        xy = np.loadtxt('data-titanic-train.csv', delimiter=',', dtype=np.str)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return  self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataSet = TitanicDataSet()
train_loader = DataLoader(dataset=dataSet, batch_size=64, shuffle=True, num_workers=10)
print(xy)




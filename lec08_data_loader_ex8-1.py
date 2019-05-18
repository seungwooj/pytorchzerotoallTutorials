# Exercise 8-1 - 같이 해봤으면 좋겠음.
# 1. Build DataLoader for Titanic data set.
# 2. Build a classifier using the DataLoader.

import torch
import torch.nn as nn
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader


class TitanicDataSet(Dataset):

    def __init__(self):
        with open("data-titanic-train.csv", "r", encoding="utf-8") as raw:
            read_line = raw.readlines()
            self.x_data = []
            self.y_data = []
            self.len = len(read_line) - 1
            for i, line in enumerate(read_line[1:]):
                replace_name = re.sub('\".+\"', "name", line)
                split_line = replace_name.replace("\n", "").split(",")
                survived = int(self.check_nullity((split_line[1])))
                p_class = int(self.check_nullity(split_line[2]))
                if split_line[4] == "male":
                    sex = 0
                elif split_line[4] == "female":
                    sex = 1
                else:
                    sex = -1
                age = float(self.check_nullity(split_line[5]))
                sibsp = int(self.check_nullity(split_line[6]))
                parch = int(self.check_nullity(split_line[7]))
                fare = float(self.check_nullity(split_line[9]))
                self.x_data.append([p_class, sex, age, sibsp, parch, fare])
                self.y_data.append(survived)

    def check_nullity(self, text):
        if len(text) == 0:
            return "0"
        else:
            return text

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataSet = TitanicDataSet()
print(dataSet.len)
train_loader = DataLoader(dataset=dataSet, batch_size=64, shuffle=True, num_workers=10)

print(train_loader.dataset.x_data)
print("")
print(train_loader.dataset.y_data)





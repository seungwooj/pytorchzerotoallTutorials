import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class DiabetesDataSet(Dataset):
    # Initialize data and divide into x_data and y_data
    def __init__(self):
        xy = np.loadtxt('data-diabetes.csv', delimiter=',', dtype= np.float32)
        self.len = xy.shape[0]  # which dimension to hand over to len
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataSet = DiabetesDataSet()
train_loader = DataLoader(dataset=dataSet, batch_size=32, shuffle=True, num_workers=2)


class ModelSigmoid(nn.Module):

    def __init__(self):
        super(ModelSigmoid, self).__init__()
        layers = []
        for i in range(len(dim_list) - 1):
            layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            layers.append(nn.Sigmoid())

        self.layer = nn.Sequential(*layers)

    def forward(self, seq):
        y = self.layer(seq)
        return y


dim_list = [8, 6, 4, 1]
model = ModelSigmoid()

# Construct loss function and optimizer
criterion = nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # Forward pass : Compute predicted y by passing x to the model
        y_pred = model(inputs)

        # Compute and print Loss
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.data)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

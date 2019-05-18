# exercise 7-1 : Classifying Diabetes with deep nets

# 1. use more than 10 layers
# 2. Use different activation function and compare the results

import numpy as np
import torch
import torch.nn as nn

# give data using numpy
xy = np.loadtxt('data-diabetes.csv', delimiter=',', dtype= np.float32)

x_data = torch.from_numpy(xy[:, 0:-1])
y_data = torch.from_numpy(xy[:, [-1]])

# check data shape
print(x_data.shape)  # [759, 8]
print(y_data.shape)  # [759, 1]


# first model : use sigmoid function as a cost function
class ModelSigmoid(nn.Module):

    def __init__(self, dim_list):
        super(ModelSigmoid, self).__init__()
        layers = []
        for i in range(len(dim_list) - 1):
            layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            layers.append(nn.Sigmoid())

        self.layer = nn.Sequential(*layers)

    def forward(self, seq):
        y = self.layer(seq)
        return y


# second model : use ReLU function as a cost function
class ModelRelu(nn.Module):

    def __init__(self, dim_list):
        super(ModelRelu, self).__init__()
        layers = []
        for i in range(len(dim_list - 1)):
            layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            layers.append(nn.ReLU())

        self.layer = nn.Sequential(*layers)

    def forward(self, seq):
        y = self.layer(seq)
        return y


dim_list = [8, 100, 90, 80, 70, 60, 50, 40, 30, 20, 1]
model1 = ModelSigmoid(dim_list)
model2 = ModelRelu()

criterion = nn.BCELoss(reduction='mean')
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)

for epoch in range(100):
    y_pred1 = model1(x_data)  # forward
    loss1 = criterion(y_pred1, y_data)
    print(epoch, loss1.data)  # model1 : 결과 0.6453

    # zero_grad : optimizer를 initialize함
    optimizer1.zero_grad()
    loss1.backward()  # backward
    optimizer1.step()  # update

print('\n')

for epoch in range(100):
    y_pred2 = model2(x_data)  # forward
    loss2 = criterion(y_pred2, y_data)
    print(epoch, loss2.data)  # model2 : 결과 0.6452

    # zero_grad : optimizer를 initialize함
    optimizer2.zero_grad()
    loss2.backward()  # backward
    optimizer2.step()  # update

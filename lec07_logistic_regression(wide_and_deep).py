import numpy as np
import torch
import torch.nn as nn

# give data using numpy
xy = np.loadtxt('data-diabetes.csv', delimiter=',', dtype=np.float32)

x_data = torch.from_numpy(xy[:, 0:-1])
y_data = torch.from_numpy(xy[:, [-1]])

# check data shape
print(x_data.shape)  # [759, 8]
print(y_data.shape)  # [759, 1]


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        layers = []
        for i in range(len(dim_list) - 1):
            layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            layers.append(nn.Sigmoid())

        self.layer = nn.Sequential(*layers)

    def forward(self, seq):
        y = self.layer(seq)
        return y


dim_list = [8, 6, 4, 1]
model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    y_pred = model(x_data)  # forward

    loss = criterion(y_pred, y_data)
    print(epoch, loss.data)
    # zero_grad : optimizer를 initialize함
    optimizer.zero_grad()
    loss.backward()  # backward
    optimizer.step()  # update
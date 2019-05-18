import numpy as np
import torch

# give data using numpy
xy = np.loadtxt('data-diabetes.csv', delimiter=',', dtype=np.float32)

x_data = torch.from_numpy(xy[:, 0:-1])
y_data = torch.from_numpy(xy[:, [-1]])

# check data shape
print(x_data.shape)  # [759, 8]
print(y_data.shape) # [759, 1]


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # use three layers to define forward function.
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y = self.sigmoid(self.l3(out2))
        return y


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
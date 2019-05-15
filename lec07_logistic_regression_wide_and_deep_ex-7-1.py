# exercise 7-1 : Classifying Diabetes with deep nets

# 1. use more than 10 layers
# 2. Use different activation function and compare the results

import numpy as np
impogrt torch

# give data using numpy
xy = np.loadtxt('data-diabetes.csv', delimiter=',', dtype=np.float32)

x_data = torch.from_numpy(xy[:, 0:-1])
y_data = torch.from_numpy(xy[:, [-1]])

# check data shape
print(x_data.shape)  # [759, 8]
print(y_data.shape)  # [759, 1]


# first model : use sigmoid function as a cost function


class Model_sigmoid(torch.nn.Module):
    def __init__(self):
        super(Model_sigmoid, self).__init__()
        # build 10 layers
        self.layer1 = torch.nn.Linear(8, 100)
        self.layer2 = torch.nn.Linear(100, 90)
        self.layer3 = torch.nn.Linear(90, 80)
        self.layer4 = torch.nn.Linear(80, 70)
        self.layer5 = torch.nn.Linear(70, 60)
        self.layer6 = torch.nn.Linear(60, 50)
        self.layer7 = torch.nn.Linear(50, 40)
        self.layer8 = torch.nn.Linear(40, 30)
        self.layer9 = torch.nn.Linear(30, 20)
        self.layer10 = torch.nn.Linear(20, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # forward using all the 10 layers
        out1 = self.sigmoid(self.layer1(x))
        out2 = self.sigmoid(self.layer2(out1))
        out3 = self.sigmoid(self.layer3(out2))
        out4 = self.sigmoid(self.layer4(out3))
        out5 = self.sigmoid(self.layer5(out4))
        out6 = self.sigmoid(self.layer6(out5))
        out7 = self.sigmoid(self.layer7(out6))
        out8 = self.sigmoid(self.layer8(out7))
        out9 = self.sigmoid(self.layer9(out8))
        y = self.sigmoid(self.layer10(out9))
        return y

# second model : use Rectifier as a cost function


class Model_ReLU(torch.nn.Module):
    def __init__(self):
        super(Model_ReLU, self).__init__()
        # build 10 layers
        self.layer1 = torch.nn.Linear(8, 100)
        self.layer2 = torch.nn.Linear(100, 90)
        self.layer3 = torch.nn.Linear(90, 80)
        self.layer4 = torch.nn.Linear(80, 70)
        self.layer5 = torch.nn.Linear(70, 60)
        self.layer6 = torch.nn.Linear(60, 50)
        self.layer7 = torch.nn.Linear(50, 40)
        self.layer8 = torch.nn.Linear(40, 30)
        self.layer9 = torch.nn.Linear(30, 20)
        self.layer10 = torch.nn.Linear(20, 1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # forward using all the 10 layers
        out1 = self.relu(self.layer1(x))
        out2 = self.relu(self.layer2(out1))
        out3 = self.relu(self.layer3(out2))
        out4 = self.relu(self.layer4(out3))
        out5 = self.relu(self.layer5(out4))
        out6 = self.relu(self.layer6(out5))
        out7 = self.relu(self.layer7(out6))
        out8 = self.relu(self.layer8(out7))
        out9 = self.relu(self.layer9(out8))
        y = self.relu(self.layer10(out9))
        return y


model1 = Model_sigmoid()
model2 = Model_ReLU()

criterion = torch.nn.BCELoss(reduction='mean')
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

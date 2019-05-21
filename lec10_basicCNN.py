import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Training settings
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST Data Sets
train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transforms)
test_dataset = datasets.MNIST(root='../data', train=False, transform=transforms)


# Data Loader (pipeline)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# Build CNN model
class Net(nn.Module):
    def __init__(self, dim_list):
        super(Net, self).__init__()
        layers = []
        for i in range(len(dim_list) - 1):
            # Modeling fully connected layers - Pooling layers - Conv layers
            layers.append(nn.ReLU(
                nn.MaxPool2d(
                    nn.Conv2d(dim_list[i], dim_list[i + 1], kernel_size=5)
                )))

        self.layer = nn.Sequential(*layers)
        self.fc = nn.Linear(lin_input, lin_output)

    def forward(self, seq):
        y = self.fc(self.layer(seq))
        return nn.functional.log_softmax(y)


# values for parameters in the model
dim_list = [1, 10, 20]
lin_input = 320
lin_output = 10

model = Net(dim_list)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# Run training cycle
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)  # TypeError 발생의 원인으로 생각되는 부분
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


# Check model accuracy
def test():
    model.test()  # model.eval() ?
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)  # TypeError 발생의 원인으로 생각되는 부분
        # sum up batch loss
        test_loss += criterion(output, target, size_average=False).data[0]
        # get the index of the max Log-probability
        pred = torch.max(output.data, 1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
          format(test_loss, correct, len(test_loader.dataset),
                 100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    test()



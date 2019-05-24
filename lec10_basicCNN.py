import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as f
from torch.utils.data import DataLoader


# Training settings
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST Data Sets
train_dataSet = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
test_dataSet = datasets.MNIST(root='../data', train=False, transform=transform)


# Data Loader (pipeline)
train_loader = DataLoader(dataset=train_dataSet, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataSet, batch_size=batch_size, shuffle=True)


# Build CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = f.relu(self.mp(self.conv1(x)))
        x = f.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)
        x = self.fc(x)
        return f.log_softmax(x, dim=1)  # avoid deprecation : write dimension for softmax


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# Run training cycle
def train(epoch):
    model.train()
    total = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            present = batch_idx * len(data)
            progress = 100. * batch_idx / len(train_loader)
            updated_loss = loss.data.item()
            print(f'Train Epoch: {epoch} [{present}/{total} ({progress:.0f}%]'
                  f'\tLoss: {updated_loss:.5f}')


# Check model accuracy
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).data.item()
        # get the index of the max Log-probability
        pred = torch.max(output.data, 1)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    total = len(test_loader.dataset)
    updated_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({updated_accuracy:.0f}%)\n')


for epoch in range(1, 10):
    train(epoch)
    test()

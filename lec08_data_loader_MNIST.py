# Dataset Loading 연습 - MNIST

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 1. torchvision.transforms
# MNIST datasets의 transform을 정의
# torchvision.transforms 를 활용하여 다양하게 data 변형이 가능
# mean= , std= 기준으로 Normalize 가능
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0,), (1.0,))
])

# 2. torchvision.datasets
# MNIST Dataset을 download할 수 있음
train_dataset = datasets.MNIST(root='./data/',
                               transform=mnist_transform,
                               train=True,
                               download=True)

valid_dataset = datasets.MNIST(root='./data/',
                               transform=mnist_transform,
                               train=False,
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              transform=mnist_transform,
                              train=False)

print(len(train_dataset), len(valid_dataset), len(test_dataset))

# 3. torch.utils.data.DataLoader
# Using Data Loader to load datasets
batch_size = 64  # option값의 정의
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

valid_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# batch_index에 대해 data와 target을 tuple형으로 return받음
for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx % 10 == 0:
        print(data.shape, target.shape)
        print(len(train_loader.dataset))
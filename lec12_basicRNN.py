import torch
import torch.nn as nn
import sys
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as f
from torch.utils.data import DataLoader

idx2char = ['h', 'i', 'e', 'l', 'o']

# Teach hihell -> ihello
x_data = [0, 1, 0, 2, 3, 3]  # hihell
one_hot_lookup = [[1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 1]]
y_data = [1, 0, 2, 3, 3, 4]  # ihello
x_one_hot = [one_hot_lookup[x] for x in x_data]

# One cell RNN input_dim (5) -> output_dim (5). sequence : 1, batch : 1
# rank = (1, 5, 5)
inputs = torch.Tensor(x_one_hot)
labels = torch.Tensor(y_data)

# Define parameters
num_classes = 5  # h, i, e, l, o
input_size = 5
hidden_size = 5
batch_size = 1  # do one by one
sequence_length = 1
num_layers = 1  # one-layer RNN


# Create RNN model
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x, hidden):
        # Reshape input in (batch_size, sequence_length, input_size)
        x = x.view(batch_size, sequence_length, input_size)  # 이 순서로 parameter 할당

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, num_classes)  # output shape을 확실하게 규
        return hidden, out

    def init_hidden(self):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        return torch.zeros(num_layers, batch_size, hidden_size)


# Instantiate RNN model
model = Model()

# Set Loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden()
    sys.stdout.write("predicted string: ")

    for input, label in zip(inputs, labels):
        # print(input.size(), label.size())
        hidden, output = model(input, hidden)
        val, idx = output.max(1)
        sys.stdout.write(idx2char[idx.data])
        loss += criterion(output, label)

    print(", epoch: %d, oss: %1.3f" % (epoch + 1, loss.data[0]))

    loss.backward()
    optimizer.step()


#
#
# loss = 0
# hidden = Model.init_hidden()
#
# sys.stdout.write("predicted string: ")
# for input, label in zip(inputs, labels):
#     hidden, output = model.
#
#
# cell = nn.RNN(input_size, hidden_size, batch_first=True)
#
# # multiple batches, one sequence input (5 inputs)
# inputs = torch.Tensor([[h, e, l, l, o], [e, o, l, l, l], [l, l, e, e, l]])  # batch_size = 3
# print("input_size", inputs.size())  # input_size torch.Size([3, 5, 4])f
#
# # initialize the hidden state.
# # (num_layers * num_directions, batch, hidden_size)
# hidden = torch.randn((1, 3, 2))
#
# # Feed to one element at a time.
# # after each step, hidden contains the hidden state.
# out, hidden = cell(inputs, hidden)
# print("out:", out.data)
# print("out_size:", out.size())  # out_size torch.Size([3, 5, 2])

# Exercise 4-5: Compute gradients using Pytorch

import torch
from torch.autograd import Variable

# give data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# a random guess for value of w1, w2, b
w1 = Variable(torch.Tensor([1.0]), requires_grad=True)
w2 = Variable(torch.Tensor([1.0]), requires_grad=True)
b = 1.0


# our hypothesis for the linear model
def forward(x):
    return (x * x) * w2 + x * w1 + b


# cost(loss) function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# Before training
print("predict (before training)", "4 hours", forward(4).data)
# Training loop
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        loss = loss(x_val, y_val)
        loss.backward()
        print("\tgrad: ", x_val, y_val, w1.grad.data, w2.grad.data)
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data

        # Manually zero the gradients after updating weights
        w1.grad.data.zero_()
        w2.grad.data.zero_()

    print("progress:", epoch, loss.data)

# After Training
print("predict (after training)", 4, forward(4).data)

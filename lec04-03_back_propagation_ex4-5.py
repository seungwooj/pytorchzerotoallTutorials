# Exercise 4-5: Compute gradients using Pytorch

import torch
from torch.autograd import Variable

# give data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# a random guess for value of w1, w2, b
w1_initial = Variable(torch.Tensor([1.0]), requires_grad=True)
w2_initial = Variable(torch.Tensor([1.0]), requires_grad=True)
b_initial = 1.0


# our hypothesis for the linear model
def forward(x, w1, w2, b):
    return (x * x) * w2 + x * w1 + b


# cost(loss) function
def calculate_loss(x, y, w1, w2, b):
    y_pred = forward(x, w1, w2, b)
    return (y_pred - y) * (y_pred - y)


# Before training
print("predict (before training)", "4 hours", forward(4, w1_initial, w2_initial, b_initial).data)
# Training loop
loss = 0
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        loss = calculate_loss(x_val, y_val, w1_initial, w2_initial, b_initial)
        loss.backward()
        print("\tgrad: ", x_val, y_val, w1_initial.grad.data, w2_initial.grad.data)
        w1_initial.data = w1_initial.data - 0.01 * w1_initial.grad.data
        w2_initial.data = w2_initial.data - 0.01 * w2_initial.grad.data

        # Manually zero the gradients after updating weights
        w1_initial.grad.data.zero_()
        w2_initial.grad.data.zero_()

    print("progress:", epoch, loss.data)

# After Training
print("predict (after training)", 4, forward(4, w1_initial, w2_initial, b_initial).data)

import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)


# model forward pass
def forward(x):
    return x * w


# loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# Before training
print("predict (before training)", 4, forward(4).data[0])

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        loss = loss(x_val, y_val)
        loss.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data)
        w.data = w.data - 0.01 * w.grad.data

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print("progress:", epoch, loss.data)

# After Training
print("predict (after training)", 4, forward(4).data)

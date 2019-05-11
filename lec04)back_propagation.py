import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w_initial = Variable(torch.Tensor([1.0]), requires_grad=True)


# model forward pass
def forward(x, w):
    return x * w


# loss function
def calculate_loss(x, y, w):
    y_pred = forward(x, w)
    return (y_pred - y) * (y_pred - y)


# Before training
print("predict (before training)", 4, forward(4, w_initial))

# Training loop
loss = 0
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        loss = calculate_loss(x_val, y_val, w_initial)
        print(loss)
        loss.backward()
        print("\tgrad: ", x_val, y_val, w_initial.grad.data)
        w_initial.data = w_initial.data - 0.01 * w_initial.grad.data

        # Manually zero the gradients after updating weights
        w_initial.grad.data.zero_()

    print("progress:", epoch, loss.data)

# After Training
print("predict (after training)", 4, forward(4, w_initial))

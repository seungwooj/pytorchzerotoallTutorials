import torch
import torch.nn.functional as F

# give data
x_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.Tensor([[0.], [0.], [1.], [1.]])


# 1. Define model
class Model(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y = F.sigmoid(self.linear(x))
        return y


model = Model(1, 1)

# 2. Construct loss and optimizer
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. Training Loop
for epoch in range(1000):
    y_pred = model.forward(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data)
    loss.backward()
    optimizer.step()

# After training
hour_var = torch.Tensor([1.0])
print("predict 1 hour", 1.0, "True" if (model.forward(hour_var).data > 0.5).item() == 1 else "False")
hour_var = torch.Tensor([7.0])
print("predict 7 hour", 7.0, "True" if (model.forward(hour_var).data > 0.5).item() == 1 else "False")









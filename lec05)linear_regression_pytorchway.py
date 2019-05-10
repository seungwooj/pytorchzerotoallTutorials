
# Inclass Example

import torch
from torch.autograd import Variable

# give data
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

# 1. Define model
class Model(torch.nn.Module):#nn.Module을 하는 이유?
    def __init__(self):
        super(Model, self).__init__()
        # initialize two nn.Linear modules in total.
        self.linear = torch.nn.Linear(1, 1)  # 1 input (x) and 1 output(y)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
model = Model() #parameter로 원래 아무것도 안 넣는 것인가?

# 2. Construct loss and optimizer
# MSELoss : linear regression의 loss function과 같은 기능 수행 library
criterion = torch.nn.MSELoss(size_average=False)
# SGD : Stochastic Gradient Descent, optimizer를 update하는 library
# model.parameter() : 앞서 선언한 model의 parameter를 변수로 받음 - update의 대상
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. Training : forward, loss, backward, step(update)
# Training loop : SGD - batch를 활용, 여러 data를 한번에 모델에 넣고 variable을 업데이트
for epoch in range(500):
    y_pred = model(x_data) # compute predicted y by passing all the x to the model
    # Forward pass : compute loss
    loss = criterion(y_pred, y_data)
    # Zero gradient?
    optimizer.zero_grad() # <- why?
    # Backward pass
    loss.backward()
    print(epoch, loss.data)
    # Update variables of Model
    optimizer.step()

# After training
hour_var = Variable(torch.Tensor([[4.0]])) #4시간 지정
print("predict (after training)", 4, model.forward(hour_var).data)

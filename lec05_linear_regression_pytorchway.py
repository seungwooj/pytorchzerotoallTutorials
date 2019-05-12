
# In-class Example

import torch

# give data
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# 1. Define model
class Model(torch.nn.Module):  # nn.Module을 하는 이유? torch의 neural network의 모든 module을 Model클래스가 상속받음
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        # initialize two nn.Linear modules in total.
        self.linear = torch.nn.Linear(input_dim, output_dim)  # 1 input (x) and 1 output(y)

    def forward(self, x):
        y = self.linear(x)
        return y


model = Model(1, 1)

# 2. Construct loss and optimizer
# MSELoss : linear regression의 loss function과 같은 기능 수행 library (cost function 에 parameter까지 받아서 criterion으로 구현
criterion = torch.nn.MSELoss(size_average=False)

# SGD : Stochastic Gradient Descent, optimizer를 update하는 library
# model.parameter() : 앞서 선언한 model의 parameter를 변수로 받음 - update의 대상
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 3. Training : forward, loss, backward, step(update)
# Training loop : SGD - batch를 활용, 여러 data를 한번에 모델에 넣고 variable을 업데이트
for epoch in range(500):
    print(x_data)
    y_pred = model(x_data)  # model클래스의 forward함수를 자동 호출
    # Forward pass : compute loss
    loss = criterion(y_pred, y_data)
    # Zero gradient?
    optimizer.zero_grad()  # <- why? optimizer를 initialize함
    # Backward pass
    loss.backward()
    print(epoch, loss.data)
    # Update variables of Model
    optimizer.step()

# After training
hour_var = torch.Tensor([[4.0]])  # 4시간 지정
print("predict (after training)", 4, model(hour_var).data)

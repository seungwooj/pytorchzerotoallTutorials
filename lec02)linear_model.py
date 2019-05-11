import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


w = 1.0  # a random guess for value of w


# our hypothesis for the linear model
def forward(x):
    return x * w


# cost(loss) function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# plotting 하기 위한 empty list 선언
w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):  # x_data, y_data 를 한쌍씩 매칭
        y_pred_val = forward(x_val)
        loss = loss(x_val, y_val)
        l_sum += loss
        print("\t", x_val, y_val, y_pred_val, loss)
    print("MSE=", l_sum / 3)

    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel("Loss")
plt.xlabel("w")
plt.show()

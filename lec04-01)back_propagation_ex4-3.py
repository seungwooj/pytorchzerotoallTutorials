# exercise 4-3 : implement computational graph and backprop using Numpy - same as ex3.0


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w_initial = 1


# our hypothesis for the linear model
def forward(x, w):
    return x * w


# cost(loss) function
def calculate_loss(x, y, w):
    y_pred = forward(x, w)
    return (y_pred - y) * (y_pred - y)


# compute gradient
def calculate_gradient(x, y, w):
    return 2 * x * (x * w - y)


# Before training
print("predict (before training)", 4, forward(4, w_initial))
# Training loop
loss = 0
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = calculate_gradient(x_val, y_val, w_initial)
        w_update = w_initial - 0.01 * grad
        print("\tgrad: ", x_val, y_val, grad)
        loss = loss(x_val, y_val)
        w_initial = w_update

# After training
print("predict (after training)", 4, forward(4, w_initial))

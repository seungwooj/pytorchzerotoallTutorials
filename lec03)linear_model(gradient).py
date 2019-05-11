# give data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# our hypothesis for the linear model
def forward(x, w):
    return x * w


# cost(loss) function
def loss(x, y, w):
    y_pred = forward(x, w)
    return (y_pred - y) * (y_pred - y)


# compute gradient
def gradient(x, y, w):
    return 2 * x * (x * w - y)


w_initial = 1.0

# Before training
print("predict (before training)", "4 hours", forward(4, w_initial))
# Training loop
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val, w_initial)
        w_new = w_initial - 0.01 * grad
        print("\tgrad: ", x_val, y_val, grad)
        l = loss(x_val, y_val, w_new)
        w_initial = w_new
    print("progress:", epoch, "w=", w_initial, "loss=", l)
# After training
print("predict (after training)", "4 hours", forward(4, w_initial))

# give data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# a random guess for value of w
w = 1.0

# our hypothesis for the linear model
def forward(x):
    return x * w
# cost(loss) function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)
# compute gradient
def gradient(x, y):
    return 2 * x * (x * w - y)

# Before training
print("predict (before training)", "4 hours", forward(4))
# Training loop
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print("\tgrad: ", x_val, y_val, grad)
        l = loss(x_val, y_val)
    print("progress:", epoch, "w=", w, "loss=", l)
# After training
print("predict (after training)", "4 hours", forward(4))

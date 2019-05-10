# Exercise 4-4: Compute gradients using computational graph (manually) - same as ex3.2

# give data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# a random guess for value of w1, w2, b
w1 = 1.0
w2 = 1.0
b = 1.0

# our hypothesis for the linear model
def forward(x):
    return np.sqrt(x) * w2 + x * w1 + b
# cost(loss) function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)
# compute gradients
def gradient_w1(x, y):
    return 2 * x * (w1 * x + w2 * (x * x) - y + b)
def gradient_w2(x, y):
    return 2 * (x * x) * (w2 * (x * x) + w1 * x - y + b)

# Before training
print("predict (before training)", "4 hours", forward(4))
# Training loop
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad_w1 = gradient_w1(x_val, y_val)
        grad_w2 = gradient_w2(x_val, y_val)
        w1 = w1 - 0.01 * grad_w1
        w2 = w2 - 0.01 * grad_w2
        print("\tgrad_w1, grad_w2: ", x_val, y_val, grad_w1, grad_w2)
        l = loss(x_val, y_val) #find loss func value using updated w1 & w2
    print("progress:", epoch, "w1=", w1, "w2=", w2, "loss=", l)
# After training
print("predict (after training)", "4 hours", forward(4))

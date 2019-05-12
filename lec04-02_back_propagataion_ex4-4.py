# Exercise 4-4: Compute gradients using computational graph (manually) - same as ex3.2

# give data
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# a random guess for value of w1, w2, b
w1_initial = 1.0
w2_initial = 1.0
b_initial = 1.0


# our hypothesis for the linear model
def forward(x, w1, w2, b):
    return (x * x) * w2 + x * w1 + b


# cost(loss) function
def calculate_loss(x, y, w1, w2, b):
    y_pred = forward(x, w1, w2, b)
    return (y_pred - y) * (y_pred - y)


# compute gradients
def calculate_gradient_w1(x, y, w1, w2, b):
    return 2 * x * (w1 * x + w2 * (x * x) - y + b)


def calculate_gradient_w2(x, y, w1, w2, b):
    return 2 * (x * x) * (w2 * (x * x) + w1 * x - y + b)


# Before training
print("predict (before training)", "4 hours", forward(4, w1_initial, w2_initial, b_initial))
# Training loop
loss = 0
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad_w1 = calculate_gradient_w1(x_val, y_val, w1_initial, w2_initial, b_initial)
        grad_w2 = calculate_gradient_w2(x_val, y_val, w1_initial, w2_initial, b_initial)
        w1_update = w1_initial - 0.01 * grad_w1
        w2_update = w2_initial - 0.01 * grad_w2
        print("\tgrad_w1, grad_w2: ", x_val, y_val, grad_w1, grad_w2)
        # find loss func value using updated w1 & w2
        loss = calculate_loss(x_val, y_val, w1_update, w2_update, b_initial)
        w1_initial = w1_update
        w2_initial = w2_update
    print("progress:", epoch, "w1=", w1_initial, "w2=", w2_initial, "loss=", loss)


# After training
print("predict (after training)", "4 hours", forward(4, w1_initial, w2_initial, b_initial))

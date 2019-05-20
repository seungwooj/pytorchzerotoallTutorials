import numpy as np
import torch
import torch.nn as nn

# Cross entropy example

Y = np.array([1, 0, 0])

Y_pred1 = np.array([0.7, 0.2, 0.1])
Y_pred2 = np.array([0.1, 0.3, 0.6])
print("Cross entropy example")
print("loss1 = ", np.sum(-Y * np.log(Y_pred1)))
print("loss2 = ", np.sum(-Y * np.log(Y_pred2)), "\n")

# Cross entropy in PyTorch

# Softmax + CrossEntropy (LogSoftmax + NLLLoss)
loss = nn.CrossEntropyLoss()

# target is size n Batch
# each element in target has to have 0 <= value < nClasses (0-2)
# Input is class, not one-hot
Y = torch.LongTensor([0])

# input is size nBatch * nClasses = 1 * 3
# Y_pred are logits (not softmax)
Y_pred1 = torch.Tensor([[2.0, 1.0, 0.1]])
Y_pred2 = torch.Tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred1, Y)  # nn.CrossEntropyLoss()하면서 [-1,0] range로 맞춰주는 게 아닌가?
l2 = loss(Y_pred2, Y)

print("Cross entropy in PyTorch")
print("PyTorch Loss1 = ", l1.data, "\nPyTorch Loss2 = ", l2.data)

Y = torch.LongTensor([2, 0, 1])  # 3 x 1
Y_pred1 = torch.Tensor([[0.1, 0.2, 0.9],
                        [1.1, 0.1, 0.2],
                        [0.2, 2.1, 0.1]])   # 3 x 3

Y_pred2 = torch.Tensor([[0.8, 0.2, 0.3],
                        [0.2, 0.3, 0.5],
                        [0.2, 0.2, 0.5]])   # 3 x 3
l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("Batch Loss1 = ", l1.data, "\nBatch Loss2 = ", l2.data)

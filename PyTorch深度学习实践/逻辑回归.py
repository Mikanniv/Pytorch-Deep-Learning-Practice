import torch.nn
# torch中的functional包含有各种激活函数
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

x_train = torch.Tensor([[1.0], [2.0], [3.0]])
y_train = torch.Tensor([[0], [0], [1]])
model = LogisticRegressionModel()
Loss = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_train)
    loss = Loss(y_pred, y_train)
    if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1}, loss: {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.xlabel('Hours')
plt.ylabel('Pass Probability')
plt.grid()
plt.show()
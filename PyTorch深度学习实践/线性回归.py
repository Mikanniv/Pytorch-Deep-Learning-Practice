import torch
class LinearModel(torch.nn.Module):
    # 构造函数
    def __init__(self):
        # 调用父类的构造函数
        super().__init__()
        # linear对象由torch.nn.Linear类实例化
        self.linear = torch.nn.Linear(1, 1)
    # 定义前向传播
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
model = LinearModel()

# 损失函数：均方根损失函数(MSE)
Loss = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = Loss(y_pred, y_data)
    if (epoch + 1) % 100 == 0:
        print(f'epoch: {epoch + 1}, loss:{loss}')
    # 梯度归零
    optimizer.zero_grad()
    loss.backward()
    # 更新权重
    optimizer.step()

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_pred = model(x_test)
print('y_pred=', y_pred.data)
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 4)
        self.fc2 = torch.nn.Linear(4, 1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class data_loader(Dataset):
    def __init__(self):
        data = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32)
        x = data[:, :-1]
        y = data[:, [-1]]
        # 简单的标准化
        mean = x.mean(axis=0)
        std = x.std(axis=0)+1e-8
        x = (x-mean)/std
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.len = data.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len

model = Model()
dataset = data_loader()
train_data = DataLoader(dataset, batch_size=32, shuffle=True)
Loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

loss_list = []
for epoch in range(100):
    loss_sum = 0
    for i, (x, y) in enumerate(train_data, 0):
        # 正向传播
        logits = model(x)
        loss = Loss(logits, y)
        loss_sum += loss.item() * x.size(0)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_avg = loss_sum / len(dataset)
    loss_list.append(loss_avg)
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss: {loss_avg:.4f}')

epoch = np.linspace(0, 100, 100)
plt.plot(epoch, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
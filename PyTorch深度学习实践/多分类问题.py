import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST("/Users/irises/PycharmProjects/RL", is_train, transform=to_tensor, download=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 64
epochs = 10
train_data = get_data_loader(is_train=True)
test_data = get_data_loader(is_train=False)
model = Model().to(device)
Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
loss_list = []

def train(epoch):
    running_loss = 0.0
    for (iteration, (x, y)) in enumerate(train_data):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x.view(-1, 28*28))
        loss = Loss(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # 每300次迭代输出一次损失
        if (iteration+1)%300==0:
            print(f'epoch:{epoch+1}, iteration:{iteration+1}, loss:{running_loss/300:.4f}')
            loss_list.append(running_loss/300)
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)
            outputs = model(x.view(-1, 28*28))
            _, predicted = torch.max(outputs.data, dim=1)
            total += y.size(0)
            correct += (predicted==y).sum().item()
    print(f'Accuracy on test set:{100*correct/total:.4f}%')

def draw():
    x, y = zip(*[(i*300, loss) for i, loss in enumerate(loss_list, start=1)])
    plt.plot(x, y)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch)
        test()
    draw()
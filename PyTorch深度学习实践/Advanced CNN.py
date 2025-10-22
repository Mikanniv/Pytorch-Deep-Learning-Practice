import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import GoogLeNet

torch.manual_seed(42)

epochs = 10
batch_size = 64

class InceptionA(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 定义分支
        self.branch_pool = torch.nn.Conv2d(in_channels, 24, 1)

        self.branch1x1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(16, 24, kernel_size=5, padding='same')

        self.branch3x3_1 = torch.nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(16, 24, kernel_size=3, padding='same')
        self.branch3x3_3 = torch.nn.Conv2d(24, 24, kernel_size=3, padding='same')

    def forward(self, x):
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        return torch.cat((branch1x1, branch5x5, branch3x3, branch_pool), 1)

class ResBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding='same')
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding='same')
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x+y)

class GoogLeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(1408, 10)
    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x

class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)
        self.rblock1 = ResBlock(16)
        self.rblock2 = ResBlock(32)

        self.fc = torch.nn.Linear(512, 10)
    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('/Users/irises/PycharmProjects/RL', is_train, transform=to_tensor, download=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')
# model = GoogLeNet().to(device)
model = ResNet().to(device)
Loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
train_data = get_data_loader(True)
test_data = get_data_loader(False)

def train(epoch):
    running_loss = 0.0
    for batch, (x, y) in enumerate(train_data):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = Loss(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss +=loss.item()
        if (batch+1)%300==0:
            print(f'epoch:{epoch+1},batch:{batch+1},loss:{running_loss/300:.3f}')

accuracy_list = []

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted==y).sum().item()
            total += y.size(0)
        accuracy = 100*correct/total
        accuracy_list.append(accuracy)
    print(f'Accuracy:{accuracy:.4f}%[{correct}/{total}]')

def draw():
    plt.plot(range(1, epochs+1), accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Epochs(ResNet)')
    plt.grid()
    plt.show()
    # plt.savefig('Advanced_CNN.png')

def main():
    for epoch in range(epochs):
        train(epoch)
        test()
    draw()

if __name__ == '__main__':
    main()
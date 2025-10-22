import  torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import  DataLoader
torch.manual_seed(42)
batch_size = 64
epochs = 10

# subsampling下采样+convolution卷积=feature extraction特征提取
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 5)
        self.conv2 = torch.nn.Conv2d(10, 20, 5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.relu(self.pooling(self.conv1(x)))
        x = self.relu(self.pooling(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('/Users/irises/PycharmProjects/RL', is_train, transform=to_tensor, download=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')
model = Net().to(device)
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

        running_loss += loss.item()
        if (batch+1)%300==0:
            print(f'epoch:{epoch+1},batch:{batch+1},average loss:{running_loss/300:.4f}')
            running_loss = 0.0

accuracy_list = []

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, dim=1)
            total +=y.size(0)
            correct += (predicted==y).sum().item()
    accuracy = 100*correct/total
    accuracy_list.append(accuracy)
    print(f'Accuracy:{accuracy:.4f}%[{correct}/{total}]')

def draw():
    plt.plot(range(1, epochs+1), accuracy_list, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Epochs')
    plt.grid()
    plt.show()
    plt.savefig('Basic_CNN.png')

if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch)
        test()
    draw()
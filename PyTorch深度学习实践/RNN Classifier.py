import torch
import numpy as np
import time
import math
import pandas as pd
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        """
        :param input_size: 输入数据维度
        :param hidden_size: 记忆体维度
        :param output_size: 输出数据维度
        :param n_layers: 层数
        :param bidirectional: 是否双向
        Embedding:嵌入层，将数据从input_size维度转换到hidden维度
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*self.n_directions, output_size)
    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers*self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)
    def forward(self, input, seq_lengths):
        # batch*seq->seq*batch
        input = input.t()
        batch_size = input.size(1)
        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)
        gru_input =  pack_padded_sequence(embedding, seq_lengths)
        output, hidden = self.gru(gru_input, hidden)
        if self.n_directions == 2:
            hidden_cat = torch.cat((hidden[-1], hidden[-2]), dim=1)
        else:
            hidden_cat = hidden[-1]
        fc_output = self.fc(hidden_cat)
        return fc_output

def create_tensor(tensor):
    tensor = tensor.to(device)
    return tensor

# 将每个名字转换为ASCII码序列
def name2list(name):
    arr = [ord(c) for c in name]
    return arr, len(arr)

def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    # 获取名字序列
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    # 获取每个名字序列的长度
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    countries = countries.long()

    # 初始化矩阵(名字总数*最长的名字序列长度)
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        # 填充前seq_len个元素
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]
    return create_tensor(seq_tensor), create_tensor(seq_lengths), create_tensor(countries)

class get_data_loader(Dataset):
    def __init__(self, is_train=True):
        filename = 'names_test.csv' if is_train else 'names_test.csv'
        rows = pd.read_csv(filename).values.tolist()
        self.names = [row[0] for row in rows]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]
        self.country_list = list(sorted((set(self.countries))))
        self.country_dict = self.getCountryDict()
        self.country_num = len(self.country_list)
    # 获取一个人的名字及其国家
    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]
    def __len__(self):
        return self.len
    # 构建国家字典
    def getCountryDict(self):
        countries_dict = dict()
        for idx, country_name in enumerate(self.country_list):
            countries_dict[country_name] = idx
        return countries_dict
    # 获取index对应的国家名字
    def idx2country(self, idx):
        return self.country_list[idx]
    # 获取国家的总数
    def getCountriesNum(self):
        return self.country_num

def time_since(since):
    s = time.time() - since
    m = math.floor(s/60)
    s -=m*60
    return '%dm %ds' % (m, s)

def train():
    total_loss = 0
    for i, (names, countries) in enumerate(train_loader, 1):
        optimizer.zero_grad()
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i%10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch}', end='')
            print(f'[{i*len(inputs)}/{len(trainset)}]', end='')
            print(f'loss={total_loss/(i*len(inputs))}')
    return total_loss

def test():
    correct = 0
    total = len(testset)
    print('Testing...')
    with torch.no_grad():
        for i, (names, countries) in enumerate(test_loader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        percent = '%.2f'%(100 * correct/total)
        print(f'Accuracy {correct}/{total} {percent}%')
    return correct/total

def draw(acc_list):
    epoch = np.arange(1, len(acc_list)+1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()

N_EPOCHS = 100
BATCH_SIZE = 256
N_LAYERS = 2
N_CHARS = 127
HIDDEN_SIZE = 100

trainset = get_data_loader(is_train=True)
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)
testset = get_data_loader(is_train=False)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
N_COUNTRY = trainset.getCountriesNum()
if __name__ == '__main__':
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYERS)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()
    print('Training for %d epoches...' % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS+1):
        train()
        acc = test()
        acc_list.append(acc)
    draw(acc_list)
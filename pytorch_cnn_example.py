import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

from utils.data_loader import get_mnist_data

# The torch version is 2.5.1.
# Train time: 58.18 s
# Test score: 98.58 %

IMAGE_H, IMAGE_W, INPUT_CHANNELS = 28, 28, 1
NUMBER_OF_CLASSES = 10
BATCH_SIZE = 64
N_EPOCHS = 10


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        base_width = 8

        self.conv1 = nn.Conv2d(1, base_width, 3, 1, 1)
        self.conv2 = nn.Conv2d(base_width, base_width * 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(base_width * 2, base_width * 4, 3, 1, 1)
        self.conv4 = nn.Conv2d(base_width * 4, base_width * 8, 3, 1, 1)
        self.conv5 = nn.Conv2d(base_width * 8, NUMBER_OF_CLASSES, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv5(x)
        x = torch.flatten(x, 1)

        x = F.log_softmax(x, dim=1)

        return x


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_mnist_data()

    X_train = X_train.reshape(-1, 1, IMAGE_H, IMAGE_W)
    X_test = X_test.reshape(-1, 1, IMAGE_H, IMAGE_W)

    print('X_train.shape', X_train.shape)
    print('X_test.shape', X_test.shape)
    print('y_train.shape', y_train.shape)
    print('y_test.shape', y_test.shape)

    print('-' * 60)
    print('The torch version is {}.'.format(torch.__version__))

    device = torch.device("cpu")

    model = Net().to(device)

    print(model)

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    y_train = y_train.type(torch.LongTensor)

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    t0 = time.time()
    optimizer = optim.Adam(model.parameters())
    model.train()
    for epoch in range(0, N_EPOCHS):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        print('Train Epoch: {} Loss: {:.6f}'.format(epoch, loss.item()))
    print("Train time: %.2f s" % (time.time() - t0))

    model.eval()
    with torch.no_grad():
        X_test = torch.Tensor(X_test)
        y_pred = model(X_test).argmax(dim=1)
        y_pred = y_pred.cpu().detach().numpy()

        score = accuracy_score(y_test, y_pred)
        print("Test score: %.2f %%" % (score * 100.0))
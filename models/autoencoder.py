from torch import optim
from torch import nn as nn
from torch.autograd import Variable
from torch.utils import data
from torchvision import datasets
from torchvision import transforms

#超参数
epochs = 10
batch_size = 64
lr = 0.005
n_test_img = 5

class AutoEncoder(nn.Module):

    def __init(self):
        super(AutoEncoder, self).__init__()

        # 压缩
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)
        )

        # 解压
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

img_transform = transforms.Compose([
    transforms.ToTensor()
])

if __name__ == '__main__':

    train_data = datasets.MNIST(root='./data', train=True,transform=img_transform, download=True)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    autoencoder = AutoEncoder()

    # 训练模型
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
    loss = nn.MSELoss()

    for epoch in range(epochs):
        for setp, (x, y) in enumerate(train_loader):
            b_x = Variable(x.view(-1, 28 * 28))
            b_y = Variable(x.view(-1, 28 * 28))
            b_label = Variable(y)
            encoded, decoded = autoencoder(b_x)
            loss_data = loss(decoded, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

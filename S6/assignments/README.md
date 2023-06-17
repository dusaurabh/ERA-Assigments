
```class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, 3)
        self.batchnorm6 = nn.BatchNorm2d(1024)
        self.conv7 = nn.Conv2d(1024, 10, 3)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool1(F.relu(self.batchnorm2(self.conv2(x))))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = self.pool2(F.relu(self.batchnorm4(self.conv4(x))))
        x = F.relu(self.batchnorm5(self.conv5(x)))
        x = F.relu(self.batchnorm6(self.conv6(x)))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 10)
        x = self.dropout(x)
        return F.log_softmax(x)
```

I have used 7 conv layers and 5 batch normalization layers, 2 max pool layers and 1 dropout layers. I have also used Fully Connected layers. I have used Relu activation function in hidden layers with padding of 1 at conv layers

At the output layer, i have used softmax activation function to predict the multi class labels


```!pip install torchsummary
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```

Here we are seeing the summary of our NN model

```torch.manual_seed(1)
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=False, **kwargs)
```

Here we are loading the training and test dataset. We are performaing normalzation at both training and testing dataloader. But only at train loader we are setting shuffle=True and we have set shuffle=False at testing loader


```from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```

Here we are doing the training and testing of model
At training, we are finding backward function and setting zero_grad() to clear out the gradients but at testing we don't find the backward function

```model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 20):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```

Here we have set the SGD gradient optimizer and learning rate of 0.01 for epochs of 19

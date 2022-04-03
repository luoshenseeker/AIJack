import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from aijack.collaborative import FedAvgClient, FedAvgServer
from aijack.utils import NumpyDataset, loadConfig

# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Batch Size

batch_size = 100
n_iters = 30000


# Model Class
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim=28 * 28, output_dim=10):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

input_dim = 28 * 28
output_dim = 10
lr_rate = 0.001

model = LogisticRegression(input_dim, output_dim)

# Instantiate the Loss Class
criterion = torch.nn.CrossEntropyLoss()  # computes softmax and then the cross entropy

# Instantiate the Optimizer Class
optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)


def prepare_dataloaders():
    try:
        at_t_dataset_train = torchvision.datasets.MNIST(
            root="./data", train=True, download=False, transform=transforms.ToTensor()
        )
        at_t_dataset_test = torchvision.datasets.MNIST(
            root="./data", train=False, download=False, transform=transforms.ToTensor()
        )
    except:
        at_t_dataset_train = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        )
        at_t_dataset_test = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transforms.ToTensor()
        )

    global_trainset = at_t_dataset_test
    global_trainloader = torch.utils.data.DataLoader(
        global_trainset, batch_size=batch_size, shuffle=True
    )
    trainset_1 = at_t_dataset_train
    trainloader_1 = torch.utils.data.DataLoader(
        trainset_1, batch_size=batch_size, shuffle=True
    )

    # return X, y, [trainloader_1, trainloader_2], global_trainloader, [half_num, tot_num - half_num]
    return [trainloader_1], global_trainloader


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print(device)

    config = loadConfig("/home/shengy/luoshenseeker/AIJack/config.yaml", True)
    print(config)
    batch_size = config['para']['batch_size']

    trainloaders, global_trainloader = prepare_dataloaders()

    criterion = nn.CrossEntropyLoss()
    client_num = 1
    adversary_client_id = 1

    net_1 = LogisticRegression()
    client_1 = FedAvgClient(net_1, user_id=0)
    client_1.to(device)
    optimizer_1 = optim.SGD(
        client_1.parameters(),
        lr=config['para']['client_1']['lr'],
        weight_decay=float(config['para']['client_1']['weight_decay']),
        momentum=config['para']['client_1']['momentum']
    )

    net_2 = LogisticRegression()
    client_2 = FedAvgClient(net_2, user_id=1)
    client_2.to(device)
    optimizer_2 = optim.SGD(
        client_2.parameters(),
        lr=config['para']['client_2']['lr'],
        weight_decay=float(config['para']['client_2']['weight_decay']),
        momentum=config['para']['client_2']['momentum']
    )

    clients = [client_1, client_2]
    optimizers = [optimizer_1, optimizer_2]

    global_model = LogisticRegression()
    global_model.to(device)

    if config['update_type'] == 'fedAVG':
        server = FedAvgServer(clients, global_model)
    else:
        server = FedAvgServer(clients, global_model)

    for epoch in range(config['para']['epoch']):
        for client_idx in range(client_num):
            client = clients[client_idx]
            trainloader = trainloaders[client_idx]
            optimizer = optimizers[client_idx]

            running_loss = 0.0
            for _, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = Variable(inputs.view(-1, 28 * 28))
                labels = Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = client(inputs)
                loss = criterion(outputs, labels.to(torch.int64))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(
                f"epoch {epoch}: client-{client_idx+1}",
                running_loss / 60000,
            )

        server.update()
        server.distribtue()

        in_preds = []
        in_label = []
        with torch.no_grad():
            for data in global_trainloader:
                inputs, labels = data
                inputs = inputs.to(device)
                inputs = Variable(inputs.view(-1, 28 * 28))
                outputs = server.server_model(inputs)
                in_preds.append(outputs)
                in_label.append(labels)
            in_preds = torch.cat(in_preds)
            in_label = torch.cat(in_label)
        print(
            f"epoch {epoch}: accuracy is ",
            accuracy_score(
                np.array(torch.argmax(in_preds, axis=1).cpu()), np.array(in_label)
            ),
        )


if __name__ == "__main__":
    main()

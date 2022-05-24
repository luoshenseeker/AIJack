from cgitb import reset
from fileinput import filename
import numpy as np
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torchsummary import summary

from aijack.attack import GAN_Attack
from aijack.collaborative import FedDSSGDClient, FedDSSGDServer
from aijack.utils import NumpyDataset, loadConfig

from opacus import PrivacyEngine

from similarity_calculation import similarity_cal

# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Batch Size
batch_size = 64

import platform

sysstr = platform.system()
if sysstr == "Linux":
    # paths
    config_path = "/home/shengy/luoshenseeker/AIJack/config.yaml"
elif sysstr == "Windows":
    # paths
    config_path = "C:\\Users\\luoshenseeker\\home\\work\\科研\\new\\AIJack\\config.yaml"
else:
    config_path = "/home/shengy/luoshenseeker/AIJack/config.yaml"

import pickle


def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"output/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)

class Generator(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(Generator, self).__init__()
        # Generator Code (from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 4, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 1, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.Tanh(),
            nn.MaxPool2d(3, 3, 1),
            nn.Conv2d(32, 64, 5),
            nn.Tanh(),
            nn.MaxPool2d(3, 3, 1),
        )

        self.lin = nn.Sequential(
            nn.Linear(256, 200), nn.Tanh(), nn.Linear(200, 11), nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape((-1, 256))
        x = self.lin(x)
        return x


def prepare_dataloaders(dataset_name):
    if dataset_name == "MNIST":
        at_t_dataset_train = torchvision.datasets.MNIST(
            root="./data/", train=True, download=True
        )
        at_t_dataset_test = torchvision.datasets.MNIST(
            root="./data/", train=False, download=True
        )

    X = at_t_dataset_train.data.numpy()
    y = at_t_dataset_train.targets.numpy()

    # ToTensor：画像のグレースケール化（RGBの0~255を0~1の範囲に正規化）、Normalize：Z値化（RGBの平均と標準偏差を0.5で決め打ちして正規化）
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # idx_1 = random.sample(range(400), 200)
    # idx_2 = list(set(range(400)) - set(idx_1))
    idx_1 = np.where(y < 5)[0]
    idx_2 = np.where(y >= 5)[0]

    global_trainset = NumpyDataset(
        at_t_dataset_test.data.numpy(),
        at_t_dataset_test.targets.numpy(),
        transform=transform,
    )
    global_trainloader = torch.utils.data.DataLoader(
        global_trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    trainset_1 = NumpyDataset(X[idx_1], y[idx_1], transform=transform)
    trainloader_1 = torch.utils.data.DataLoader(
        trainset_1, batch_size=batch_size, shuffle=True, num_workers=2
    )
    trainset_2 = NumpyDataset(X[idx_2], y[idx_2], transform=transform)
    trainloader_2 = torch.utils.data.DataLoader(
        trainset_2, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return X, y, [trainloader_1, trainloader_2], global_trainloader, [200, 200]

def get_filename(args, config):
    if args.no_attack:
        no_attack = "noattack"
    else:
        no_attack = "attack"
    if args.poison:
        poi = "pp"
    else:
        poi = "np"
    if args.privacy:
        return "_".join(["dp", f"up{args.upload_p}", f"n{args.max_grad_norm}", f"d{args.target_delta}", f"e{args.target_epsilon}", poi, no_attack]) 
    else:
        return "_".join(["normal", f"up{args.upload_p}", f"gepoch{config['para']['generator']['epoch']}", poi, no_attack]) 

def main(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    print(device)


    if not os.path.exists('output'):
        print("Path doesn't exist: output")
        exit(1)

    config = loadConfig(config_path, True)
    print(config)
    file_name = get_filename(args, config)
    print(file_name)
    global batch_size
    batch_size = config['para']['batch_size']
    image_size = config['imagesize']
    n_iter = config['para']['epoch']
    simulatiry_calculate_interval = args.simulatiry_calculate_interval

    if not args.no_attack and not os.path.exists(f"output/{file_name}"):
        os.mkdir(f"output/{file_name}")

    X, y, trainloaders, global_trainloader, dataset_nums = prepare_dataloaders(config['dataset'])

    criterion = nn.CrossEntropyLoss()
    client_num = 2
    adversary_client_id = 1
    target_label = config['para']['target_label']

    net_1 = Net()
    client_1 = FedDSSGDClient(net_1, user_id=0, upload_p=args.upload_p, device=device)
    client_1.to(device)
    optimizer_1 = optim.SGD(
        client_1.parameters(), 
        lr=config['para']['client_1']['lr'], 
        weight_decay=float(config['para']['client_1']['weight_decay']), 
        momentum=config['para']['client_1']['momentum']
    )

    net_2 = Net()
    client_2 = FedDSSGDClient(net_2, user_id=1, upload_p=args.upload_p, device=device)
    client_2.to(device)
    optimizer_2 = optim.SGD(
        client_2.parameters(), 
        lr=config['para']['client_2']['lr'], 
        weight_decay=float(config['para']['client_2']['weight_decay']), 
        momentum=config['para']['client_2']['momentum']
    )

    clients = [client_1, client_2]
    optimizers = [optimizer_1, optimizer_2]

    generator = Generator(nz, nc, ngf)
    generator.to(device)
    optimizer_g = optim.SGD(
        generator.parameters(), 
        lr=config['para']['generator']['lr'], 
        weight_decay=float(config['para']['generator']['weight_decay']), 
        momentum=config['para']['generator']['momentum']
    )
    gan_attacker = GAN_Attack(
        client_2,
        target_label,
        generator,
        optimizer_g,
        criterion,
        nz=nz,
        device=device,
    )

    global_model = Net()
    global_model.to(device)

    if config['update_type'] == 'fedAVG':
        server = FedDSSGDServer(clients, global_model)

    # summary
    print("net", net_1)
    print("Generator", generator)
    if args.summary_only:
        return

    fake_batch_size = batch_size
    fake_label = config['para']['fake_label']

    loss_hist = np.zeros((n_iter + 1, client_num))
    acc_hist = np.zeros((n_iter + 1, client_num))
    similarity_score_hist = np.zeros((n_iter//simulatiry_calculate_interval, 2))

    if args.privacy:
        privacy_engines = []

        for client_idx in range(client_num):
            client = clients[client_idx]
            trainloader = trainloaders[client_idx]
            optimizer = optimizers[client_idx]
            privacy_engine = PrivacyEngine(
                client, max_grad_norm=args.max_grad_norm, 
                batch_size=batch_size, 
                target_delta=args.target_delta,
                target_epsilon=args.target_epsilon,
                epochs=n_iter,
                # noise_multiplier=args.noise_multiplier, 
                sample_size=len(trainloader)
            )
            privacy_engine.attach(optimizer)
            privacy_engines.append(privacy_engine)
        
    for epoch in range(n_iter):
        for client_idx in range(client_num):
            client = clients[client_idx]
            trainloader = trainloaders[client_idx]
            optimizer = optimizers[client_idx]

            running_loss = 0.0
            for _, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                if not args.no_attack:
                    if epoch >= 1 and client_idx == adversary_client_id:
                        fake_image = gan_attacker.attack(fake_batch_size)
                        inputs = torch.cat([inputs, fake_image])
                        labels = torch.cat(
                            [
                                labels,
                                torch.tensor([fake_label] * fake_batch_size, device=device),
                            ]
                        )

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
                running_loss / dataset_nums[client_idx],
            )

            loss_hist[epoch, client_idx] = running_loss / dataset_nums[client_idx]

        server.update()
        server.distribtue()

        if not args.no_attack:
            gan_attacker.update_discriminator()
            gan_attacker.update_generator(
                config['para']['generator']['batch_size'], 
                config['para']['generator']['epoch'], 
                config['para']['generator']['log_interval']
            )

        in_preds = []
        in_label = []
        with torch.no_grad():
            for data in global_trainloader:
                inputs, labels = data
                inputs = inputs.to(device)
                outputs = server.server_model(inputs)
                in_preds.append(outputs)
                in_label.append(labels)
            in_preds = torch.cat(in_preds)
            in_label = torch.cat(in_label)
        acc =  accuracy_score(
                np.array(torch.argmax(in_preds, axis=1).cpu()), np.array(in_label)
            )
        print(
            f"epoch {epoch}: accuracy is ",
            acc
        )
        acc_hist[epoch, 0] = acc
        acc_hist[epoch, 1] = acc

        if not args.no_attack:
            reconstructed_image = gan_attacker.attack(1).cpu().numpy().reshape(28, 28)
            print(
                "reconstrunction error is ",
                np.sqrt(
                    np.sum(
                        (
                            (X[np.where(y == target_label)[0][:10], :, :] - 0.5 / 0.5)
                            - reconstructed_image
                        )
                        ** 2
                    )
                ) / (10 * (28 * 28))
            )
            target_imgs = X[np.where(y == target_label)[0][:10], :, :]
            # similarity calculation
            if (epoch + 1) % simulatiry_calculate_interval == 0:
                atk_img = reconstructed_image
                similarity_score_mr = np.max([similarity_cal(atk_img, i, "mr") for i in target_imgs])
                similarity_score_ssim = np.max([similarity_cal(atk_img, i, "ssim") for i in target_imgs])
                print("-"*100)
                print(f"similarity: mr:{similarity_score_mr}, ssim:{similarity_score_ssim}")
                similarity_score_hist[(epoch + 1)//simulatiry_calculate_interval - 1, 0] = similarity_score_mr
                similarity_score_hist[(epoch + 1)//simulatiry_calculate_interval - 1, 1] = similarity_score_ssim
            # print(reconstructed_image)
            fig = plt.figure(frameon=False)
            fig.set_size_inches(image_size, image_size)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.imshow(reconstructed_image * 0.5 + 0.5, vmin=-1, vmax=1, cmap='gray', )
            plt.savefig(f"output/{file_name}/{epoch}.png")
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(similarity_score_hist, "similarity_score", file_name)
    if not os.system(f"zip -q output/pic/{file_name}.zip output/{file_name}/*.png"):
        os.system(f"rm -r output/{file_name}")
    else:
        print("Error in zip pictures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--privacy', dest="privacy", action='store_true', help="Implement dp.")
    parser.add_argument("-p", "--poison", dest="poison", action="store_true", help="Implement poison attack.")
    parser.add_argument("-s", "--summary_only", dest="summary_only", action="store_true", help="Show summary only.")
    parser.add_argument("--no_attack", dest="no_attack", action="store_true", help="No attack.")
    parser.add_argument("--max_grad_norm", type=float, default=1)
    parser.add_argument("--simulatiry_calculate_interval", type=int, default=20)
    # parser.add_argument("--noise_multiplier", type=float, default=1.1)
    parser.add_argument("--target_delta", type=float, default=1)
    parser.add_argument("--target_epsilon", type=float, default=100)
    parser.add_argument("--upload_p", type=float, default=1)
    args = parser.parse_args()
    print("start at:" + time.asctime(time.localtime(time.time())))
    main(args)
    print("Finish at" + time.asctime(time.localtime(time.time())))

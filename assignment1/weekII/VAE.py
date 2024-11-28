from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_mps = not args.no_mps and torch.backends.mps.is_available()

args.batch_size = 256

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=False, **kwargs)

from train import get_train_loader, get_test_loader

train_loader = get_train_loader(args.batch_size)

test_loader = get_test_loader(args.batch_size)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 1)
        self.fc22 = nn.Linear(400, 1)
        self.fc3 = nn.Linear(1, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'assign2/results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


# def task_A(epoch):
#     import matplotlib.pyplot as plt
#     """* Train a VAE model on the MNIST dataset with a 2 dimensional latent space. Plot the encoded samples in the latent space and color-code the different digit classes"""
#     model.eval()  # Ensure the model is in evaluation mode

#     latdim1, latdim2, labels = [], [], []
#     with torch.no_grad():
#         for i, (data, label) in enumerate(test_loader):
#             data = data.to(device)
#             label = label.to(device)
            
#             # Assuming model.encode outputs a single latent vector
#             latent_dim = model.encode(data.view(-1, 784))
#             latdim1.append(latent_dim[:, 0].cpu())  # First latent dimension
#             latdim2.append(latent_dim[:, 1].cpu())  # Second latent dimension
#             labels.append(label.cpu())
    
#     # Convert lists of tensors into flattened NumPy arrays
#     latdim1 = torch.cat(latdim1).numpy()
#     latdim2 = torch.cat(latdim2).numpy()
#     labels = torch.cat(labels).numpy()

#     # Plot the latent space
#     fig, ax = plt.subplots(1, 1, figsize=(7, 7))
#     scatter = ax.scatter(latdim1, latdim2, c=labels, cmap='tab10', alpha=0.7)
#     legend1 = ax.legend(*scatter.legend_elements(), title="Digits")
#     ax.add_artist(legend1)
#     ax.set_title("Latent Space Representation")
#     ax.set_xlabel("Latent Dimension 1")
#     ax.set_ylabel("Latent Dimension 2")
#     plt.show()

def task_A(epoch):
    import matplotlib.pyplot as plt
    """* Train a VAE model on the MNIST dataset with a 2-dimensional latent space. 
    Plot the encoded samples in the latent space and color-code the different digit classes."""
    
    model.eval()  # Ensure the model is in evaluation mode

    latdim1, latdim2, labels = [], [], []
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data = data.to(device)
            label = label.to(device)

            # Obtain the mean (mu) from the encoder for latent space representation
            mu, _ = model.encode(data.view(-1, 784))
            latdim1.append(mu[:, 0].cpu())  # First latent dimension
            latdim2.append(mu[:, 1].cpu())  # Second latent dimension
            labels.append(label.cpu())
    
    # Convert lists of tensors into flattened NumPy arrays
    latdim1 = torch.cat(latdim1).numpy()
    latdim2 = torch.cat(latdim2).numpy()
    labels = torch.cat(labels).numpy()

    # Plot the latent space
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    scatter = ax.scatter(latdim1, latdim2, c=labels, cmap='tab10', alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Digits")
    ax.add_artist(legend1)
    ax.set_title(f"Latent Space Representation at Epoch {epoch}")
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    plt.savefig(f'assign2/results/latent_space_epoch_{epoch}.png')
    plt.show()






if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        task_A(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 1).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'assign2/results/sample_' + str(epoch) + '.png')
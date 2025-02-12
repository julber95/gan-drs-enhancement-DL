import torch
import os
from tqdm import trange
import argparse
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from model import Generator, Discriminator
from utils import D_train, G_train, save_models, dynamic_soft_truncation
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GAN with dynamic soft truncation.')
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dataset = datasets.MNIST(root='data/MNIST/', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    G = Generator(g_output_dim=784).to(device)
    D = Discriminator(784).to(device)

    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr)

    d_losses = []
    g_losses = []

    for epoch in trange(1, args.epochs + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(-1, 784).to(device)
            D_loss = D_train(x, G, D, D_optimizer, criterion, device)

            z = torch.randn(x.size(0), 100).to(device)
            D_scores = D(G(z)).detach()
            z = dynamic_soft_truncation(z, D_scores)
            G_loss = G_train(z, G, D, G_optimizer, criterion, device)

            epoch_d_loss += D_loss
            epoch_g_loss += G_loss

        d_losses.append(epoch_d_loss / len(train_loader))
        g_losses.append(epoch_g_loss / len(train_loader))

        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{args.epochs}] - D_loss: {d_losses[-1]:.4f}, G_loss: {g_losses[-1]:.4f}")

        if epoch % 10 == 0:
            save_models(G, D, 'checkpoints')

    plt.figure()
    plt.plot(range(1, len(d_losses) + 1), d_losses, label='D_loss')
    plt.plot(range(1, len(g_losses) + 1), g_losses, label='G_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('plots/training_losses_dynamic_truncation.png')
    plt.show()

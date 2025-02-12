import torch
import torchvision
import os
import argparse
from model import Generator, Discriminator
from utils import load_model, dynamic_soft_truncation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate using dynamic soft truncation.')
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for generation.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    generator = load_model(Generator(g_output_dim=784).to(device), 'checkpoints')
    discriminator = Discriminator(d_input_dim=784).to(device)
    discriminator.load_state_dict(torch.load('checkpoints/D.pth', map_location=device))
    generator.eval()
    discriminator.eval()

    os.makedirs('samples', exist_ok=True)
    n_samples = 0

    with torch.no_grad():
        while n_samples < args.num_samples:
            z = torch.randn(args.batch_size, 100).to(device)
            D_scores = discriminator(generator(z).view(args.batch_size, -1)).squeeze().detach()
            z = dynamic_soft_truncation(z, D_scores)

            x_generated = generator(z).reshape(args.batch_size, 1, 28, 28)
            x_flattened = x_generated.view(args.batch_size, -1)

            P_accept = D_scores / (1 - D_scores + 1e-6)
            M_local = P_accept.max().item()
            P_accept /= M_local

            for i in range(args.batch_size):
                if n_samples >= args.num_samples:
                    break
                if torch.rand(1).item() < P_accept[i]:
                    torchvision.utils.save_image(
                        x_generated[i], os.path.join('samples', f'{n_samples}.png'))
                    n_samples += 1

    print(f"Dynamic soft truncation generation complete: {n_samples} samples saved in 'samples/'.")

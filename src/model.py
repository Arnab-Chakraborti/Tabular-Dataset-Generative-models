import torch
import torch.nn as nn
from torchinfo import summary

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, z):
        return self.network(z)

if __name__ == "__main__":
    model = Generator(noise_dim=14, output_dim=9)
    summary(model, input_size=(1, 14))

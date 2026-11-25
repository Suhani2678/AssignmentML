import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_dim=326, input_size=32 * 32 * 3):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        return self.classifier(self.embedding(x))


class SmallCNN(nn.Module):
    def __init__(self, embedding_dim=240):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, 10)

    def forward(self, x):
        return self.classifier(self.embedding(self.convolution(x)))

    @torch.no_grad()
    def encode(self, x):
        return self.embedding(self.convolution(x))


class ConvAutoencoder(nn.Module):
    def __init__(self, bottleneck_dim=256):
        super().__init__()
        self.encoder_convolution = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_projection = nn.Linear(64 * 4 * 4, bottleneck_dim)
        self.decoder_projection = nn.Linear(bottleneck_dim, 64 * 4 * 4)
        self.decoder_convolution = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        features = self.encoder_convolution(x).flatten(1)
        return self.encoder_projection(features)

    def forward(self, x):
        z = self.encode(x)
        x = self.decoder_projection(z).reshape(-1, 64, 4, 4)
        return self.decoder_convolution(x)

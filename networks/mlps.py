from torch import nn


class MLPs(nn.Module):
    def __init__(self, input_channel: int, output_channel: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channel, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, output_channel)
        )

    def forward(self, x):
        return self.net(x)


class MLPsActivate(nn.Module):
    def __init__(self, input_channel: int, output_channel: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channel, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, output_channel),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

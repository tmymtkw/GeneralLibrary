from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Identity()

    def forward(self, x):
        x = self.f(x)

        return x
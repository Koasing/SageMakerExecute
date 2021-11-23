from torch import nn


class SimpleModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        return x

import torch
import torch.nn as nn
import copy
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExtrapolationNetwork(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(ExtrapolationNetwork, self).__init__()

        self.fc1 = nn.Linear(in_shape, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_shape)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        return x




if __name__ == "__main__":
    net = ExtrapolationNetwork(16*5, 16)
    pass
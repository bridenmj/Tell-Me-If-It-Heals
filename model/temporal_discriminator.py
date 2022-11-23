import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


CUDA = True if torch.cuda.is_available() else False

# some parameters
N_CHANNELS = 3 # input channels (r, g, b) = 3


class SiameseCNN(nn.Module):
    # initializers
    def __init__(self, img_shape, latent_dim):
        super(SiameseCNN, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(3, 8, kernel_size = 3, stride = 1, padding=1, padding_mode="zeros") # outsize: insize + 2*padding - kernel + 1
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3, stride = 1, padding=1, padding_mode="zeros") # outsize: insize + 2*padding - kernel + 1
        #18 * 64 * 64 input features, 512 output features (see sizing flow below)

        self.fc1 = nn.Linear(int(16 * img_shape[1]//2//2 * img_shape[2]//2//2), 128)
        self.fc2 = nn.Linear(128, latent_dim)

        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2_bn = nn.BatchNorm1d(latent_dim)


    # forward method
    def forward(self, img):

        x = F.leaky_relu(self.conv1(img), 0.2)
        #x = F.tanh(self.conv1(img))
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x), 0.2)
        #x = F.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0),-1)

        x = self.fc1_bn(self.fc1(x))
        #x = F.tanh(x) #############################################################
        x = F.leaky_relu(x, 0.2)
        x = self.fc2_bn(self.fc2(x))
        #x = F.tanh(x)
        x = F.leaky_relu(x, 0.2)
        
        return x




class Classifier(nn.Module):
    # takes in CNN encodings, concats, and discriminates
    def __init__(self, latent_dim):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(latent_dim * 2, latent_dim // 2)
        self.fc2 = nn.Linear(latent_dim // 2, 1)

        self.fc1_bn = nn.BatchNorm1d(latent_dim // 2)

    def forward(self, u, v):
        z = torch.cat((u, v), dim=1)
        z = self.fc1_bn(self.fc1(z))
        #z = self.fc1(z)
        #z = F.tanh(z)
        z = F.leaky_relu(z, 0.2)
        #print(z)
        logit = self.fc2(z)
        pred = torch.sigmoid(logit)

        return pred



class TemporalDiscriminator(nn.Module):
    # previously known as TemporalClassifier
    def __init__(self, img_shape, latent_dim=32):
        super(TemporalDiscriminator, self).__init__()
        self.latent_dim = latent_dim

        self.cnn = SiameseCNN(img_shape, latent_dim)
        self.classifier = Classifier(latent_dim)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.1)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, img_1, img_2):
        # img_1 should be a predecessor of img_2
        u = self.cnn(img_1)
        v = self.cnn(img_2)
        pred = self.classifier(u,v)

        return pred

        
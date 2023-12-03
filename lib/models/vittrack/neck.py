import torch
from torch import nn
import torch.nn.functional as F


class NECK_FPN(nn.Module):
    def __init__(self, input_dim, output_dim, num_x=256, num_layers=3, BN=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_x = num_x
        self.num_layers = num_layers
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(input_dim, input_dim), nn.BatchNorm1d(input_dim))
                                        for _ in range(num_layers - 1))
        else:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(input_dim, input_dim)) for _ in range(num_layers - 1))
        self.proj1 = nn.Linear(input_dim, output_dim)
        self.proj2 = nn.Linear(input_dim, output_dim)

    def forward(self, xz_list):
        global_vector = xz_list[-1][:, 0:1, :]
        xz_list_3 = [xz_list[3], xz_list[7], xz_list[11]]
        fb_features = []
        for i in range(len(xz_list_3)):
            x = xz_list_3[i][:, 1:self.num_x+1, :]
            fb_features.append(x)
        x = fb_features[-1]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = x + fb_features[-2-i]
            if i < self.num_layers - 1:
                x = F.relu(x)
        x = self.proj1(x)
        global_vector = self.proj2(global_vector)
        xz = torch.cat((global_vector, x), dim=1)
        return xz

import torch
import torch.nn.functional as F
from lib import graph_functional
import lib.augmentation
from torch.nn import Sequential, Linear, ReLU


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, edge_index_, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index_)

        z, g = self.encoder(x, edge_index, batch)
        # print(z) # 7549 * 64
        # print(g) # 128 * 64
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2


class Aug_Encoder(torch.nn.Module):
    def __init__(self, input_dim, encoder, args):
        super(Aug_Encoder, self).__init__()
        self.device = args.device
        self.encoder = encoder
        self.mask = torch.rand(input_dim, 1, requires_grad=True, device=self.device)
        self.mlp_edge_model = Sequential(Linear(input_dim, 1),
                                         ReLU(), Linear(1, input_dim), ReLU()).to(self.device)

    def forward(self, edge_index):
        #mask = torch.sigmoid(self.mask[:edge_index.shape[1], :])
        mask = torch.sigmoid(
            torch.transpose(self.mlp_edge_model(torch.transpose(self.mask, 0, 1)), 0, 1)[:edge_index.shape[1], :])
        return mask
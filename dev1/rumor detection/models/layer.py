import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            #nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, z_dim * 2),
        )

    def forward(self, x):
        params = self.net(x) # N * 4
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:] # N * 2
        sigma = softplus(sigma) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)


class AmbiguityLearning(nn.Module):
    def __init__(self):
        super(AmbiguityLearning, self).__init__()
        # self.encoding = EncodingPart()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()

    def forward(self, text_encoding, image_encoding):
        p_z1_given_text = self.encoder_text(text_encoding)
        p_z2_given_image = self.encoder_image(image_encoding)
        z1 = p_z1_given_text.rsample()
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1)/ 2.
        skl = torch.sigmoid(skl)
        return skl


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channels // reduction, channels * 2, 1, bias=False),
        )

    def forward(self, x):
        w = F.adaptive_avg_pool1d(x, 1)  # Squeeze
        w = self.fc(x)
        w, b = w.split(w.data.size(1) // 2, dim=1)  # Excitation
        w = torch.sigmoid(w)

        return x * w + b  # Scale and add bias

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.conv_lower = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

        self.conv_upper = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(channels)
        )

        self.se_block = SEBlock(channels)

    def forward(self, x):
        path = self.conv_lower(x)
        path = self.conv_upper(path)

        path = self.se_block(path)

        path = x + path
        return F.relu(path)


class MultiMessageFusion(nn.Module):
    def __init__(self, in_channel, filters, blocks, num_classes):
        super(MultiMessageFusion, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channel, filters, 3, padding=1, bias=False),
            nn.BatchNorm1d(filters),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(*[ResBlock(filters) for _ in range(blocks - 1)])

        self.out_conv = nn.Sequential(
            nn.Conv1d(filters, 128, 1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        # print("after conv:{}".format(x.shape))
        x = self.res_blocks(x)
        # print("after res:{}".format(x.shape))
        x = self.out_conv(x)
        # print("after out_conv:{}".format(x.shape))
        x = F.adaptive_avg_pool1d(x, 1)
        # print("after avg pool:{}".format(x.shape))

        x = x.view(x.data.size(0), -1)
        # print("after view:{}".format(x.shape))
        x = self.fc(x)

        return F.log_softmax(x, dim=1)
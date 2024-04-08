import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForMaskedLM

import math


class FastCNN(nn.Module):
    # a CNN-based altertative approach of bert for text encoding
    def __init__(self, channel=32, kernel_size=(1, 2, 4, 8)):
        super(FastCNN, self).__init__()
        self.fast_cnn = nn.ModuleList()
        for kernel in kernel_size:
            self.fast_cnn.append(
                nn.Sequential(
                    nn.Conv1d(200, channel, kernel_size=kernel),
                     #kernel_size*in_channel e.g. 1*200,2*200
                    # (N, 200, 30) -> (N, 32, 30-2 +1)
                    nn.BatchNorm1d(channel),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.AdaptiveMaxPool1d(1) # (N,32,1)
                )
            )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CLIP(nn.Module):
    def __init__(self, out_channels,
        cnn_channel=32,
        cnn_kernel_size=(1, 2, 4, 8),
        shared_image_dim=128,
        shared_text_dim=128):
        super(CLIP, self).__init__()

        self.shared_image = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_image_dim),
            # nn.BatchNorm1d(shared_image_dim),
            nn.ReLU()
        )
        self.img_fc = nn.Linear(shared_image_dim, out_channels)

        self.shared_text_encoding = FastCNN(
            channel=cnn_channel,
            kernel_size=cnn_kernel_size
        )
        self.shared_text_linear = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )
        ntoken, ninp, nhead, nhid, nlayers, dropout = 49408, 768, 8, 2048, 12, 0.5
        self.encoder = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.text_fc = nn.Linear(ninp, out_channels)

    def forward(self, img, text):
        n_batch = img.size(0)
        img = img.float()
        img_out = self.shared_image(img)
        img_out = self.img_fc(img_out)
        img_out = F.normalize(img_out, p=2, dim=-1)

        text_encoding = self.shared_text_encoding(text)
        text_shared = self.shared_text_linear(text_encoding)
        text_shared = text_shared.long()
        src = self.encoder(text_shared) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        text_out = self.transformer_encoder(src, None)

        text_out = text_out[:, -1, :]
        text_out = self.text_fc(text_out)
        text_out = F.normalize(text_out, p=2, dim=-1)

        return img_out, text_out

    def encode_image(self, image):
        n_batch = image.size(0)

        out = self.img_model(image)
        out = self.avg_pool(out)
        out = out.view(n_batch, -1)
        out = self.img_fc(out)

        return out

    def encode_text(self, text):
        src = self.encoder(text) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src, None)

        out = out[:, -1, :]
        out = self.text_fc(out)

        return out
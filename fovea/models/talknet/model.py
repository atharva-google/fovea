"""TalkNet audio-visual model: encoders + cross/self attention + classifier head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from .audio_encoder import audioEncoder
from .visual_encoder import visualConv1D, visualFrontend, visualTCN


class attentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, tar):
        src = src.transpose(0, 1)  # B, T, C -> T, B, C
        tar = tar.transpose(0, 1)  # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None, key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src.transpose(0, 1)  # T, B, C -> B, T, C


class talkNetModel(nn.Module):
    # NOTE: attribute names must match the checkpoint keys - don't rename them.
    def __init__(self):
        super().__init__()
        # Visual temporal encoder
        self.visualFrontend = visualFrontend()
        self.visualTCN = visualTCN()
        self.visualConv1D = visualConv1D()

        # Audio temporal encoder
        self.audioEncoder = audioEncoder(layers=[3, 4, 6, 3], num_filters=[16, 32, 64, 128])

        # Audio-visual cross attention
        self.crossA2V = attentionLayer(d_model=128, nhead=8)
        self.crossV2A = attentionLayer(d_model=128, nhead=8)

        # Audio-visual self attention
        self.selfAV = attentionLayer(d_model=256, nhead=8)

        self.criterion = nn.CrossEntropyLoss()
        self.FC = nn.Linear(256, 2)

    def lossAV(self, x, labels=None):
        x = x.squeeze(1)
        x = self.FC(x)
        if labels is None:
            predScore = x[:, 1]
            predScore = predScore.t()
            return predScore.view(-1).detach().cpu().numpy()
        nloss = self.criterion(x, labels)
        predScore = F.softmax(x, dim=-1)
        predLabel = torch.round(F.softmax(x, dim=-1))[:, 1]
        correctNum = (predLabel == labels).sum().float()
        return nloss, predScore, predLabel, correctNum

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape
        x = x.view(B * T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)
        x = x.transpose(1, 2)
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        return x.transpose(1, 2)

    def forward_audio_frontend(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        return self.audioEncoder(x)

    def forward_cross_attention(self, x1, x2):
        x1_c = self.crossA2V(src=x1, tar=x2)
        x2_c = self.crossV2A(src=x2, tar=x1)
        return x1_c, x2_c

    def forward_audio_visual_backend(self, x1, x2):
        x = torch.cat((x1, x2), 2)
        x = self.selfAV(src=x, tar=x)
        return torch.reshape(x, (-1, 256))

    def forward_audio_backend(self, x):
        return torch.reshape(x, (-1, 128))

    def forward_visual_backend(self, x):
        return torch.reshape(x, (-1, 128))

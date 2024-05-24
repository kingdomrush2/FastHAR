import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch import Tensor
from torch.nn import MultiheadAttention
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_seq_len=120):
        super(PositionalEncoding, self).__init__()
        self.pos_enc = self.positional_encoding(hidden_size, max_seq_len)

    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1)].cuda()
        return x

    def positional_encoding(self, hidden_size, max_seq_len):
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pos_enc = torch.zeros(max_seq_len, hidden_size)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.nhead, self.d_k)
        k = self.linear_k(k).view(batch_size, -1, self.nhead, self.d_k)
        v = self.linear_v(v).view(batch_size, -1, self.nhead, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == True, -1e9)

        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.out(output)

        return output, attention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True, dropout=dropout)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):

        src2, attn_map = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_map


class TimeSeriesTransformer(nn.Module):
    def __init__(self, channel_size, d_model, nhead, num_layers, dim_feedforward, seq_length, output_size, is_trans_pt,
                 is_mask, filter_pos):
        super(TimeSeriesTransformer, self).__init__()
        self.is_trans_pt = is_trans_pt
        self.channel_size = channel_size
        self.is_mask = is_mask
        self.filter_pos = filter_pos

        self.embedding = nn.Linear(int(seq_length / 2), d_model)

        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, nhead, dim_feedforward) for _ in range(num_layers)])

        self.bn = nn.BatchNorm1d(num_features=channel_size)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = torch.fft.rfft(x, dim=2)
        x = x[:, :, 1:]

        amplitude = torch.abs(x)

        fundamental_freq_index = torch.argmax(amplitude, axis=2)
        if self.is_trans_pt:
            Cut_off_frequency = fundamental_freq_index + (
                        self.filter_pos * (x.size(2) * torch.ones((x.size(0), x.size(1)))
                                           - fundamental_freq_index) / 4).int()
        else:
            Cut_off_frequency = fundamental_freq_index + (
                        self.filter_pos * (x.size(2) * torch.ones((x.size(0), x.size(1))).cuda()
                                           - fundamental_freq_index) / 4).int()

        for i in range(Cut_off_frequency.size(0)):
            for j in range(Cut_off_frequency.size(1)):
                amplitude[i, j, Cut_off_frequency[i, j] + 1:] = 0

        x = self.embedding(amplitude)
        x = torch.relu(self.bn(x))

        if self.is_trans_pt:
            src_mask = torch.zeros(self.channel_size, self.channel_size).type(torch.bool)
        else:
            src_mask = torch.zeros(self.channel_size, self.channel_size).type(torch.bool).cuda()
        mask_pos = [[0, 4], [0, 5], [1, 3], [1, 5], [2, 3], [2, 4], [3, 1], [3, 2], [4, 0], [4, 2], [5, 0], [5, 1]]
        for i in range(len(mask_pos)):
            src_mask[mask_pos[i][0], mask_pos[i][1]] = True

        for layer in self.layers:
            if self.is_mask:
                x, _ = layer(x, src_mask=src_mask)
            else:
                x, _ = layer(x)

        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x


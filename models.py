import torch
from torch import nn


class TemporalGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalGCN, self).__init__()
        self.gcn = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, channels, seq_len = x.size()
        adj_matrix = torch.eye(seq_len).to(x.device)
        x = x.permute(0, 2, 1)
        x = torch.matmul(adj_matrix, x)
        x = self.gcn(x)
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        return self.relu(x)

class MultiScaleDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleDilatedConv, self).__init__()
        self.dilated_conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dilated_conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        self.dilated_conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=8, dilation=8)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        res = x if self.residual is None else self.residual(x)
        x1 = self.bn1(self.dilated_conv1(x))
        x2 = self.bn2(self.dilated_conv2(x))
        x3 = self.bn3(self.dilated_conv3(x))
        return x1 + x2 + x3 + res

class FeatureFusion(nn.Module):
    def __init__(self, channels):
        super(FeatureFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, gcn_features, msdc_features):
        fusion = torch.cat((gcn_features, msdc_features), dim=1)
        attention_weights = self.attention(fusion)
        return gcn_features * attention_weights + msdc_features * (1 - attention_weights)

class TemporalDilatedGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalDilatedGCNBlock, self).__init__()
        self.temporal_gcn = TemporalGCN(in_channels, out_channels)
        self.multi_scale_dilated_conv = MultiScaleDilatedConv(in_channels, out_channels)
        self.fusion = FeatureFusion(out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        gcn_features = self.temporal_gcn(x)
        msdc_features = self.multi_scale_dilated_conv(x)
        fused_features = self.fusion(gcn_features, msdc_features)
        return self.bn(fused_features)

# Multi-Head Attention Module
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.embed_size, bias=False)
        self.keys = nn.Linear(self.head_dim, self.embed_size, bias=False)
        self.queries = nn.Linear(self.head_dim, self.embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # Query * Key

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))  # Apply mask if provided

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # Scaled dot-product attention

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.embed_size
        )

        out = self.fc_out(out)  # Linear transformation of concatenated heads
        return out

class StackedBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(StackedBiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        # 强制将LSTM权重调整为连续的内存块
        self.bilstm.flatten_parameters()

        # 打印LSTM输入形状
        # print(f"LSTM input shape: {x.shape}")

        out, _ = self.bilstm(x)

        # 打印LSTM输出形状
        # print(f"LSTM output shape: {out.shape}")

        return out

# TSB Module
class TSB(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, heads, output_dim):
        super(TSB, self).__init__()

        self.encoder_bilstm = StackedBiLSTM(input_dim, hidden_dim, num_layers)
        self.decoder_bilstm = StackedBiLSTM(hidden_dim * 2, hidden_dim, num_layers)

        self.multihead_attention = MultiHeadAttention(hidden_dim * 2, heads)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encoder
        encoder_out = self.encoder_bilstm(src)
        encoder_out = self.norm(encoder_out)

        # Apply Multi-Head Attention to Encoder output
        attn_out = self.multihead_attention(encoder_out, encoder_out, encoder_out, mask=src_mask)
        attn_out = self.norm(attn_out)

        # Decoder
        decoder_out = self.decoder_bilstm(attn_out)
        decoder_out = self.norm(decoder_out)

        out = self.fc_out(decoder_out)
        return out

# Main Model
class ecgTransForm(nn.Module):
    def __init__(self, configs, hparams):
        super(ecgTransForm, self).__init__()

        self.temporal_dilated_gcn = TemporalDilatedGCNBlock(configs.input_channels, configs.mid_channels)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, hparams["feature_dim"], kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(hparams["feature_dim"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(0.2)

        # Replace Transformer with TSB
        self.tsb = TSB(input_dim=hparams["feature_dim"], hidden_dim=hparams["feature_dim"], num_layers=2, heads=8,
                       output_dim=hparams["feature_dim"])

        self.crm = self._make_layer(SEBasicBlock, hparams["feature_dim"], 3)
        self.aap = nn.AdaptiveAvgPool1d(1)

        # 更新线性层的输入尺寸，使其匹配池化后的特征维度
        self.clf = nn.Linear(94, configs.num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv1d(planes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(planes),
        )
        layers = [block(planes, planes, downsample=downsample)]
        for _ in range(1, blocks):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x_in):
        x = self.temporal_dilated_gcn(x_in)
        x = self.dropout(x)
        x = self.conv_block2(x)

        # TSB (replaces Transformer)
        x = x.permute(0, 2, 1)  # Adjust for TSB input
        x = self.tsb(x, x)  # Using src and tgt both as x
        x = x.permute(0, 2, 1)

        x = self.crm(x).permute(0, 2, 1)
        x = self.aap(x)

        x_flat = x.view(x.size(0), -1)  # 确保 x_flat 的形状为 (batch_size, feature_dim)
        x_out = self.clf(x_flat)
        return x_out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, *, reduction=4):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

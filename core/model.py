import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        scale = self.avg_pool(x).view(batch_size, channels)
        scale = self.fc(scale).view(batch_size, channels, 1, 1)
        return x * scale


class BasicBlockSE(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)
        return out


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_channels, attention_channels=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, attention_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(attention_channels, in_channels, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        # x: [B, C, T]
        weights = self.attention(x)
        mean = torch.sum(weights * x, dim=2)
        var = torch.sum(weights * (x - mean.unsqueeze(2)).pow(2), dim=2)
        std = torch.sqrt(var.clamp_min(1e-5))
        return torch.cat([mean, std], dim=1)


class ResNet34_SE(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        input_channels=1,
        n_mels=64,
        channels=(32, 64, 128, 256),
        pooling="ASP",
    ):
        super().__init__()
        if pooling not in {"ASP", "SAP"}:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.n_mels = n_mels
        self.pooling = pooling
        self.inplanes = channels[0]

        self.input_norm = nn.InstanceNorm1d(n_mels)
        self.conv1 = nn.Conv2d(
            input_channels,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(channels[0], blocks=3, stride=1)
        self.layer2 = self._make_layer(channels[1], blocks=4, stride=2)
        self.layer3 = self._make_layer(channels[2], blocks=6, stride=2)
        self.layer4 = self._make_layer(channels[3], blocks=3, stride=2)

        feat_dim = self._infer_feature_dim(n_mels)
        self.pool = AttentiveStatsPool(feat_dim)
        pool_out_dim = feat_dim if pooling == "SAP" else feat_dim * 2
        self.pool_bn = nn.BatchNorm1d(pool_out_dim)
        self.fc = nn.Linear(pool_out_dim, embedding_dim)
        self.bn_final = nn.BatchNorm1d(embedding_dim)

        self._reset_parameters()

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlockSE(self.inplanes, planes, stride=stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlockSE(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def _forward_conv(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def _infer_feature_dim(self, n_mels):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, 200)
            dummy = self._forward_conv(dummy)
        return dummy.size(1) * dummy.size(2)

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # x: [B, 1, F, T]
        x = x.squeeze(1)
        x = self.input_norm(x)
        x = x.unsqueeze(1)

        x = self._forward_conv(x)
        batch_size, channels, freq_bins, frames = x.shape
        x = x.reshape(batch_size, channels * freq_bins, frames)

        pooled = self.pool(x)
        if self.pooling == "SAP":
            pooled = pooled[:, : x.size(1)]

        pooled = self.pool_bn(pooled)
        x = self.fc(pooled)
        x = self.bn_final(x)
        return x

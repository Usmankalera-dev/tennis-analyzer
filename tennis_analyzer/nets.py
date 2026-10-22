"""Neural network definitions for the tennis analyzer.

Two pretrained networks are used:

1. BallNet - TrackNet architecture (U-Net style). Input is 3 consecutive
   frames stacked as 9 channels at 640x360. Output is a 256-class
   quantized heatmap of the ball position. Weights come from the public
   TrackNet implementation by yastrebksv (GitHub), trained on broadcast
   tennis videos.

2. CourtNet - same backbone, 3-channel input (single frame), 15 output
   heatmaps: 14 tennis court keypoints + court center. Weights come from
   the public TennisCourtDetector project by yastrebksv (GitHub).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class _TrackNetBackbone(nn.Module):
    """Shared encoder-decoder body used by both pretrained models."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def encode_decode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        return x


class BallNet(_TrackNetBackbone):
    """TrackNet ball tracker. Input: (B, 9, 360, 640). Output with
    testing=True: (B, 256, 360*640) softmax probabilities over quantized
    heatmap values 0..255 for every pixel."""

    def __init__(self, out_channels=256):
        super().__init__(in_channels=9, out_channels=out_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, testing=False):
        batch = x.size(0)
        x = self.encode_decode(x)
        out = x.reshape(batch, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        return out


class CourtNet(_TrackNetBackbone):
    """Court keypoint detector. Input: (B, 3, 360, 640).
    Output: (B, 15, 360, 640) raw heatmaps (apply sigmoid)."""

    def __init__(self, out_channels=15):
        super().__init__(in_channels=3, out_channels=out_channels)

    def forward(self, x):
        return self.encode_decode(x)


def load_ball_net(weights_path, device="cpu"):
    model = BallNet(out_channels=256)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_court_net(weights_path, device="cpu"):
    model = CourtNet(out_channels=15)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

"""
Real modality-specific encoders and a local fusion+classifier head
(Sec. III-C). Encoders are exchanged/aggregated across vehicles; the fusion head
stays local. Used by the real multimodal FL training in sim/real_fl.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT = 64
NCLS = 3


class ImageEncoder(nn.Module):
    """Small 2D CNN encoder for the camera patch (3x32x32 -> FEAT)."""
    def __init__(self, feat=FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, feat)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class LidarEncoder(nn.Module):
    """PointNet-style encoder for box LiDAR points (P x 3 -> FEAT)."""
    def __init__(self, feat=FEAT):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )
        self.fc = nn.Linear(128, feat)

    def forward(self, x):                       # x: [B, P, 3]
        h = self.mlp(x)
        h = h.max(dim=1)[0]                      # symmetric pooling
        return self.fc(h)


class FusionHead(nn.Module):
    """Local fusion + classifier over the modality features."""
    def __init__(self, modalities, feat=FEAT, ncls=NCLS):
        super().__init__()
        self.modalities = list(modalities)
        self.head = nn.Sequential(
            nn.Linear(feat * len(self.modalities), 64), nn.ReLU(),
            nn.Linear(64, ncls),
        )

    def forward(self, feats):                   # feats: dict modality->[B,FEAT]
        z = torch.cat([feats[m] for m in self.modalities], dim=1)
        return self.head(z)


class RadarEncoder(nn.Module):
    """PointNet-style encoder for sparse box radar returns
    (P x [x, y, vx, vy, rcs] -> FEAT). Radar returns per object are few
    (often 0-5), so the encoder is deliberately small."""
    def __init__(self, feat=FEAT):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
        )
        self.fc = nn.Linear(64, feat)

    def forward(self, x):                       # x: [B, P, 5]
        h = self.mlp(x)
        h = h.max(dim=1)[0]                      # symmetric pooling
        return self.fc(h)


class RadarMapEncoder(nn.Module):
    """Small 2D CNN for FMCW range-Doppler maps (1 x 32 x 32 -> FEAT),
    e.g. DeepSense 6G radar cubes after FFT processing."""
    def __init__(self, feat=FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, feat)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class GPSEncoder(nn.Module):
    """Tiny MLP for position features (D -> FEAT)."""
    def __init__(self, feat=FEAT, din=3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 32), nn.ReLU(),
                                 nn.Linear(32, feat))

    def forward(self, x):
        return self.net(x)


_ENCODERS = {"camera": ImageEncoder, "lidar": LidarEncoder,
             "radar": RadarEncoder}
# per-run overrides, e.g. DeepSense uses a range-Doppler map radar encoder
# and a GPS encoder: ENCODER_OVERRIDES.update({"radar": RadarMapEncoder,
# "gps": GPSEncoder})
ENCODER_OVERRIDES = {}


def make_encoder(modality, feat=FEAT):
    cls = ENCODER_OVERRIDES.get(modality, _ENCODERS.get(modality))
    if cls is None:
        raise KeyError(f"no encoder registered for modality '{modality}'")
    return cls(feat)


def encoder_forward(enc, modality, img, lid, rad=None):
    if modality == "camera":
        return enc(img)
    if modality == "radar":
        return enc(rad)
    return enc(lid)

"""
Two models:

  EMGNet        -- the original architecture, reconstructed from Chen et al.
                   2020 (Sensors 20(3), 672), Figure 6 + Section 3.2 + Table
                   1. Kept as-is and validated against the paper's reported
                   parameter count (see paper_param_count_check) -- this is
                   what the professor asked to be implemented, and stays
                   here as the faithfulness reference.

  EMGNetFusion  -- EMGNet's own four-stage CWT-CNN backbone, PLUS a second
                   branch over classical hand-crafted time-domain features
                   (MAV/RMS/WL/VAR/ZC/SSC/WAMP/IEMG), fused before the final
                   classifier. This is the model actually trained by
                   run_pipeline.py. Motivation: a plain Random Forest on
                   just the TD features scored well above the pure CWT-CNN
                   on this dataset (see README) -- rather than switching
                   away from EMGNet, its own backbone is kept and extended
                   with that same signal instead of leaving it on the table.

Both use the same pre-activation (BN -> ReLU -> Conv) 3x3 conv stages,
producing 16 -> 32 -> 32 -> 64 feature maps, stride 2 exactly when the
channel count increases (per the paper's own stated rule), stride 1
otherwise, and no fully-connected layer over the raw scalogram -- only
adaptive average pooling.
"""
import torch
import torch.nn as nn

from . import config


class _BNReLUConv(nn.Module):
    """BatchNorm -> ReLU -> Conv2d, the pre-activation block used throughout EMGNet."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, bias=True)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class _EMGNetBackbone(nn.Module):
    """The four conv stages shared by EMGNet and EMGNetFusion, ending in a
    64-channel feature map (before any classifier head)."""

    def __init__(self, in_channels: int, dropout: float = 0.0):
        super().__init__()
        self.stage1 = _BNReLUConv(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.stage2 = _BNReLUConv(16, 32, kernel_size=3, stride=2, padding=1)   # channels grow -> stride 2
        self.stage3 = _BNReLUConv(32, 32, kernel_size=3, stride=1, padding=1)  # channels unchanged -> stride 1
        self.stage4 = _BNReLUConv(32, 64, kernel_size=3, stride=2, padding=1)  # channels grow -> stride 2
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.stage1(x))
        x = self.drop(self.stage2(x))
        x = self.drop(self.stage3(x))
        x = self.drop(self.stage4(x))
        return x  # (batch, 64, H', W')


class EMGNet(nn.Module):
    """The paper's own architecture: backbone + BN-ReLU-1x1Conv classifier + GAP."""

    def __init__(self, in_channels: int = config.NUM_CHANNELS,
                 num_classes: int = config.NUM_CLASSES,
                 dropout: float = 0.0):
        super().__init__()
        self.backbone = _EMGNetBackbone(in_channels, dropout)
        self.classifier = _BNReLUConv(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.classifier(x)
        x = self.pool(x)
        return x.flatten(1)  # (batch, num_classes) logits


class _TDBranch(nn.Module):
    """Small MLP over classical time-domain features -> a compact embedding."""

    def __init__(self, num_features: int, hidden: int = 64, out: int = 32, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
            nn.BatchNorm1d(out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EMGNetFusion(nn.Module):
    """
    EMGNet's CWT-CNN backbone (unchanged) + a classical TD-feature branch,
    fused before a linear classifier. This is the model run_pipeline.py
    actually trains -- see module docstring for why.
    """

    def __init__(self, in_channels: int = config.NUM_CHANNELS,
                 num_td_features: int = config.NUM_TD_FEATURES,
                 num_classes: int = config.NUM_CLASSES,
                 cnn_dropout: float = 0.0, td_dropout: float = 0.0,
                 td_hidden: int = 64, td_out: int = 32):
        super().__init__()
        self.cnn_backbone = _EMGNetBackbone(in_channels, cnn_dropout)
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)
        self.td_branch = _TDBranch(num_td_features, hidden=td_hidden, out=td_out, dropout=td_dropout)
        self.classifier = nn.Linear(64 + td_out, num_classes)

    def forward(self, x_cwt: torch.Tensor, x_td: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn_pool(self.cnn_backbone(x_cwt)).flatten(1)  # (batch, 64)
        td_feat = self.td_branch(x_td)                                 # (batch, td_out)
        fused = torch.cat([cnn_feat, td_feat], dim=1)
        return self.classifier(fused)  # (batch, num_classes) logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def paper_param_count_check() -> None:
    """Reproduces the paper's Table 1 EMGNet parameter count (34,311) for the
    paper's own 8-channel / 7-class Myo configuration -- a check that the
    EMGNet class (the faithfulness reference; EMGNetFusion is the trained
    model, see module docstring) matches the published architecture."""
    reference_model = EMGNet(in_channels=8, num_classes=7)
    n = count_parameters(reference_model)
    expected = 34311
    status = "MATCH" if n == expected else "MISMATCH"
    print(f"[paper_param_count_check] 8ch/7class EMGNet -> {n} params "
          f"(paper Table 1: {expected}) -> {status}")
    if n != expected:
        raise AssertionError(
            f"EMGNet reconstruction does not match paper's reported parameter "
            f"count ({n} != {expected}); re-check model.py against Figure 6."
        )


if __name__ == "__main__":
    paper_param_count_check()
    fusion = EMGNetFusion()
    print(f"EMGNetFusion ({config.NUM_CHANNELS}ch CNN + {config.NUM_TD_FEATURES}-dim TD / "
          f"{config.NUM_CLASSES} classes): {count_parameters(fusion)} params")

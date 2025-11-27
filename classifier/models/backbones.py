import torch
import torch.nn as nn
import torchxrayvision as xrv

class CXRBackbone(nn.Module):
    """
    Wraps a TorchXRayVision DenseNet and exposes:
      - feats: penultimate features for probes
      - logits: original 14-pathology outputs (if you want them)
    """
    def __init__(self, weights="densenet121-res224-nih", frozen=True):
        super().__init__()
        self.base = xrv.models.DenseNet(weights=weights)
        if frozen:
            for p in self.base.parameters():
                p.requires_grad = False
            self.base.eval()

        # In XRV, self.base.features is usually the penultimate representation
        # and self.base.classifier is the final pathology head.
        self.feature_dim = self.base.features.classifier.in_features \
                           if hasattr(self.base.features, "classifier") \
                           else 1024  # fallback, you can inspect in a REPL

    def forward(self, x):
        # XRV's forward returns logits, but we want features too.
        # Easiest way is to mimic its forward:
        feats = self.base.features(x)           # shape: (B, feature_dim)
        logits = self.base.classifier(feats)    # shape: (B, 14) for pathologies
        return feats, logits

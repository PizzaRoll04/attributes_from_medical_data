import torch
import torch.optim as optim
import torch.nn.functional as F
from classifier.models.backbones import CXRBackbone
from classifier.models.general_models import MLP
from classifier.datasets.nih_cxr import NIHCXRDataset

def train_epoch(loader, device, optimizer, model_probe, backbone, sensitive_y="gender"):
    # setup a frozen backbone
    backbone = CXRBackbone(frozen=True) 
    probe = MLP(backbone.feature_dim, 1) if model_probe is None else model_probe
    total_loss = 0.0

    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3) if optimizer is None else optimizer

    for batch in loader:
        imgs = batch["image"].to(device)
        age = batch["sensitive"]["age"].to(device)
        y = batch["sensitive"][sensitive_y].to(device)
        # labels_y = batch["label_y"].to(device) # this is the affliction

        optimizer.zero_grad()
        with torch.no_grad():
            features, _ = backbone(imgs)
        logits_y = probe(features).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits_y, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)

import torch
import torch.optim as optim
import torch.nn.functional as F
from classifier.models.backbones import CXRBackbone
from classifier.models.general_models import MLP

def train_epoch(loader, optimizer, device, model_probe=None):
    # setup a frozen backbone
    backbone = CXRBackbone(frozen=True)
    probe = MLP() if model_probe is None else model_probe
    total_loss = 0.0

    for batch in loader:
        imgs = batch["image"].to(device)
        tabs = batch["tabular"].to(device)
        labels_y = batch["label_y"].to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            features, _ = backbone(imgs)
        logits_y, z_joint = model(imgs, tabs)
        loss = F.binary_cross_entropy_with_logits(logits_y, labels_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)

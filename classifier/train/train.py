import torch.optim as optim
import torch.nn.functional as F

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        imgs = batch["image"].to(device)
        tabs = batch["tabular"].to(device)
        labels_y = batch["label_y"].to(device)

        optimizer.zero_grad()
        logits_y, z_joint = model(imgs, tabs)
        loss = F.binary_cross_entropy_with_logits(logits_y, labels_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)

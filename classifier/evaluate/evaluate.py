import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            tabs = batch["tabular"].to(device)
            y = batch["label_y"].cpu().numpy()

            logits, _ = model(imgs, tabs)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_true.extend(y)
            y_pred.extend(probs)

    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, (np.array(y_pred) > 0.5).astype(int))
    return {"auc": auc, "accuracy": acc}

from classifier.datasets import NIHCXRDataset
from classifier.train.train_probe import train_epoch
from classifier.models.backbones import CXRBackbone
from classifier.models.general_models import MLP

from pathlib import Path

from tqdm import tqdm
import torch
import argparse
import torchvision
import torchxrayvision as xrv
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_images", type=str, help="Directory of dataset")
    parser.add_argument("--path_csv", type=str, help="Path to dataset csv")
    parser.add_argument(
        "--num_epochs", type=int, default=30, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--target_sensitive_field",
        type=str,
        default="age",
        help="The target sensitive datapoint to probe/train against",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        default="./xrv_sensitive_out",
        help="Directory for outputs for things like model weights",
    )
    parser.add_argument(
        "--path_model",
        type=str,
        help="Path to a pretrained model to use to bootstrap training",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="*",
        default=[256, 128, 64],
        help="Hidden layer sizes for the MLP probe. "
        "Example: --hidden_dims 256 128 64. "
        "Empty → linear probe.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dir_out = Path(args.dir_out)
    dir_out = dir_out.expanduser()
    dir_out.mkdir(parents=True, exist_ok=True)
    assert dir_out.exists() == True

    hidden_dims = args.hidden_dims
    path_model = args.path_model
    lr = args.lr

    num_epochs = args.num_epochs
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )
    target = args.target_sensitive_field

    transforms = torchvision.transforms.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ]
    )  # from torchxrayvision pretrained model
    # set up data loaders
    nih_dataset_train = NIHCXRDataset(
        csv_path=args.path_csv,
        img_root=args.dir_images,
        img_transform=transforms,
        split="train",
    )
    nih_dataset_val = NIHCXRDataset(
        csv_path=args.path_csv,
        img_root=args.dir_images,
        img_transform=transforms,
        split="val",
    )

    loader_train = DataLoader(
        nih_dataset_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    loader_val = DataLoader(
        nih_dataset_val, batch_size=32, shuffle=False, num_workers=4
    )

    backbone = CXRBackbone(frozen=True)
    backbone.to(device)
    if path_model is not None:
        assert Path(path_model).exists()
        state_probe = torch.load(path_model, map_location="cpu")
        target = state_probe["target"]
        probe = MLP(
            state_probe["feature_dim"],
            1,
            dims_hidden=state_probe["hidden_dims"],
        )
        probe.load_state_dict(state_probe["probe_state_dict"])
        probe.to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        optimizer.load_state_dict(state_probe["optimizer_state_dict"])
    else:
        probe = MLP(backbone.feature_dim, 1, dims_hidden=hidden_dims)
        probe.to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    is_classification = target == "gender"
    losses_train, losses_val = [], []  # per epoch
    metrics_train, metrics_val = [], []  # per epoch
    for epoch in range(num_epochs):
        total_loss_train = 0.0
        total_examples_train = 0

        total_correct_train = 0
        squared_error_sum_train = 0.0

        train_bar = tqdm(loader_train, desc="Train", leave=False)
        for batch in train_bar:
            # training
            imgs = batch["image"].to(device)
            y = batch["sensitive"][target].to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                features, _ = backbone(imgs)
            logits_y = probe(features).squeeze(-1)

            if is_classification:
                loss = F.binary_cross_entropy_with_logits(logits_y, y)
            else:
                loss = F.mse_loss(logits_y, y)

            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            total_loss_train += loss.item() * bs
            total_examples_train += bs

            if is_classification:
                # accuracy
                probs = torch.sigmoid(logits_y)
                preds = (probs >= 0.5).float()
                correct = (preds == y).sum().item()
                total_correct_train += correct
                batch_acc = correct / bs
                train_bar.set_postfix(loss=loss.item(), acc=f"{batch_acc:.3f}")
            else:
                # RMSE components
                se = (logits_y - y) ** 2
                squared_error_sum_train += se.sum().item()
                rmse_batch = (se.mean().item()) ** 0.5
                train_bar.set_postfix(loss=loss.item(), rmse=f"{rmse_batch:.3f}")

        avg_train_loss = total_loss_train / total_examples_train
        if is_classification:
            train_metric = total_correct_train / total_examples_train  # accuracy
        else:
            train_metric = (
                squared_error_sum_train / total_examples_train
            ) ** 0.5  # RMSE

        losses_train.append(avg_train_loss)
        metrics_train.append(train_metric)

        if epoch % 2 == 0:
            path_save = dir_out / f"probe_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "probe_state_dict": probe.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "feature_dim": backbone.feature_dim,
                    "target": target,
                    "hidden_dims": probe.dims_hidden,
                },
                path_save,
            )

        # validation
        backbone.eval()
        probe.eval()

        total_loss_val = 0.0
        total_correct_val = 0.0
        total_examples_val = 0
        squared_error_sum_val = 0.0

        val_bar = tqdm(loader_val, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in val_bar:
                imgs = batch["image"].to(device, non_blocking=True)
                y = batch["sensitive"][target].to(device)

                features, _ = backbone(imgs)
                logits_y = probe(features).squeeze(-1)

                if is_classification:
                    loss = F.binary_cross_entropy_with_logits(logits_y, y)
                else:
                    loss = F.mse_loss(logits_y, y)

                bs = imgs.size(0)
                total_loss_val += loss.item() * bs
                total_examples_val += bs

                if is_classification:
                    probs = torch.sigmoid(logits_y)
                    preds = (probs >= 0.5).float()
                    correct = (preds == y).sum().item()
                    total_correct_val += correct
                    batch_acc = correct / bs
                    val_bar.set_postfix(loss=loss.item(), acc=f"{batch_acc:.3f}")
                else:
                    se = (logits_y - y) ** 2
                    squared_error_sum_val += se.sum().item()
                    rmse_batch = (se.mean().item()) ** 0.5
                    val_bar.set_postfix(loss=loss.item(), rmse=f"{rmse_batch:.3f}")

        avg_val_loss = total_loss_val / total_examples_val
        if is_classification:
            val_metric = total_correct_val / total_examples_val  # accuracy
        else:
            val_metric = (squared_error_sum_val / total_examples_val) ** 0.5  # RMSE

        losses_val.append(avg_val_loss)
        metrics_val.append(val_metric)

        # epoch summary
        if is_classification:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"train_loss={avg_train_loss:.4f}, train_acc={train_metric:.4f} | "
                f"val_loss={avg_val_loss:.4f},   val_acc={val_metric:.4f}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"train_loss={avg_train_loss:.4f}, train_RMSE={train_metric:.4f} | "
                f"val_loss={avg_val_loss:.4f},   val_RMSE={val_metric:.4f}"
            )

    path_save = dir_out / f"probe_epoch_final_{num_epochs-1}.pt"
    torch.save(
        {
            "epoch": num_epochs - 1,
            "probe_state_dict": probe.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "feature_dim": backbone.feature_dim,
            "target": target,
            "hidden_dims": probe.dims_hidden,
        },
        path_save,
    )

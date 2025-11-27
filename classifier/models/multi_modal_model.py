import torch
import torch.nn as nn
import torchvision.models as tvm

class MultiModalModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # Image encoder backbone
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        # Remove final classifier
        self.image_encoder = nn.Sequential(
            *list(backbone.children())[:-1]  # output shape: (B, 512, 1, 1)
        )
        img_feat_dim = 512

        # # Other data encoder
        # self.tab_encoder = nn.Sequential(
        #     nn.Linear(other_in_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        # )

        self.classifier_y = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # binary target_y example
        )

    def forward(self, img, tab):
        # Image features
        z_img = self.image_encoder(img)          # (B, 512, 1, 1)
        z_img = z_img.squeeze(-1).squeeze(-1)    # (B, 512)

        # Tabular features
        z_tab = self.tab_encoder(tab)            # (B, hidden_dim)

        # Joint representation
        z_joint = torch.cat([z_img, z_tab], dim=1)  # (B, 512 + hidden_dim)

        # Main-task logits
        logits_y = self.classifier_y(z_joint).squeeze(-1)  # (B,)

        return logits_y, z_joint  # also return z_joint for privacy analysis

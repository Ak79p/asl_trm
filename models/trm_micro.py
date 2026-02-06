import torch
import torch.nn as nn


class TRMMicro(nn.Module):

    def __init__(
        self,
        num_classes,
        input_dim=384,     # feature dimension per frame
        dim=128,          # model dimension (was already reduced)
        num_heads=4,
        num_z_tokens=8,
        reason_steps=4
    ):
        super().__init__()

        self.dim = dim
        self.num_z_tokens = num_z_tokens
        self.reason_steps = reason_steps

        # -------------------------
        # Input projection
        # -------------------------
        self.input_proj = nn.Linear(input_dim, dim)

        # -------------------------
        # Pre-encoder (Phase-3.1 change)
        # -------------------------
        self.pre_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 2,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=1   
        )

        # -------------------------
        # Latent Z tokens
        # -------------------------
        self.z_tokens = nn.Parameter(
            torch.randn(1, num_z_tokens, dim)
        )

        # -------------------------
        # Reasoning block (shared)
        # -------------------------
        self.reason_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            batch_first=True,
            dropout=0.1
        )

        # -------------------------
        # Post-encoder (unchanged)
        # -------------------------
        self.post_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 2,
                batch_first=True,
                dropout=0.1
            ),
            num_layers=1
        )

        # -------------------------
        # Normalization + classifier
        # -------------------------
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x):
        # Accept (B,T,D) or (B,T,J,D)
        if x.dim() == 4:
            B, T, J, D = x.shape
            x = x.view(B, T, J * D)
        elif x.dim() == 3:
            B, T, _ = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        x = self.input_proj(x)
        x = self.pre_encoder(x)

        z = self.z_tokens.expand(B, -1, -1)

        for _ in range(self.reason_steps):
            z = self.reason_layer(z)

        x = torch.cat([z, x], dim=1)
        x = self.post_encoder(x)

        x = x.mean(dim=1)
        x = self.norm(x)

        return self.classifier(x)

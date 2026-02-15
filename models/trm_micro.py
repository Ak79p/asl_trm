import torch
import torch.nn as nn


class TRMMicro(nn.Module):
    """
    TRM-Micro with SHARED GATED Local + Global Reasoning

    Local  : Z ⟲ Z        (runs 4 times)
    Global : Z ⟲ (Z+Frames)  (runs 2 times, always active)

    Uses ONE shared transformer layer for iterative reasoning.
    """

    def __init__(
        self,
        num_classes,
        input_dim=384,      # 48 joints × 8 dims
        dim=128,
        num_heads=4,
        num_z_tokens=8,
        local_steps=4,
        global_steps=2
    ):
        super().__init__()

        self.dim = dim
        self.num_z_tokens = num_z_tokens
        self.local_steps = local_steps
        self.global_steps = global_steps

        # -------------------------
        # Input projection
        # -------------------------
        self.input_proj = nn.Linear(input_dim, dim)

        # -------------------------
        # Temporal encoder (frames)
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
        # Shared Reasoning Layer
        # -------------------------
        self.reason_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            batch_first=True,
            dropout=0.1
        )

        # Learned gate (initialized slightly open)
        self.global_gate = nn.Parameter(torch.tensor(0.1))

        # -------------------------
        # Post fusion
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
        # Output
        # -------------------------
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x):

        # -------- Input handling --------
        if x.dim() == 4:
            B, T, J, D = x.shape
            x = x.view(B, T, J * D)
        elif x.dim() == 3:
            B, T, _ = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        # -------- Frame encoding --------
        x = self.input_proj(x)          # (B, T, dim)
        x = self.pre_encoder(x)         # (B, T, dim)

        # -------- Latent tokens --------
        z = self.z_tokens.expand(B, -1, -1)  # (B, Z, dim)

        # -------- Local reasoning (4x) --------
        for _ in range(self.local_steps):
            z = self.reason_layer(z)

        # -------- Global reasoning (2x always) --------
        gate = torch.tanh(self.global_gate)

        for _ in range(self.global_steps):
            zx = torch.cat([z, x], dim=1)       # (B, Z+T, dim)
            zx = self.reason_layer(zx)
            z_new = zx[:, :self.num_z_tokens]
            z = z + gate * z_new

        # -------- Final fusion --------
        x = torch.cat([z, x], dim=1)
        x = self.post_encoder(x)

        x = x.mean(dim=1)
        x = self.norm(x)

        return self.classifier(x)

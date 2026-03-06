import torch
import torch.nn as nn


class TRMMicro(nn.Module):

    def __init__(
        self,
        num_classes,
        input_dim=384,
        dim=160,              # reduced from 192
        num_heads=4,
        num_z_tokens=12,      # reduced from 16
        local_steps=4,
        global_steps=2,
        max_len=64
    ):
        super().__init__()

        self.dim = dim
        self.num_z_tokens = num_z_tokens
        self.local_steps = local_steps
        self.global_steps = global_steps
        self.max_len = max_len

        # Input projection
        self.input_proj = nn.Linear(input_dim, dim)

        # Positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, dim)
        )

        # Frame encoder (↑ dropout increased)
        self.pre_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 2,
                batch_first=True,
                dropout=0.2
            ),
            num_layers=1
        )

        # Latent tokens
        self.z_tokens = nn.Parameter(
            torch.randn(1, num_z_tokens, dim)
        )

        # Shared reasoning layer
        self.reason_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            batch_first=True,
            dropout=0.2
        )

        self.global_gate = nn.Parameter(torch.tensor(0.1))

        # Final fusion
        self.post_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 2,
                batch_first=True,
                dropout=0.2
            ),
            num_layers=1
        )

        # Attention pooling
        self.attn_pool = nn.Linear(dim, 1)

        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):

        if x.dim() == 4:
            B, T, J, D = x.shape
            x = x.view(B, T, J * D)
        elif x.dim() == 3:
            B, T, _ = x.shape
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        x = self.input_proj(x)

        if T > self.max_len:
            raise ValueError("Sequence length exceeds max_len")

        x = x + self.pos_embedding[:, :T]
        x = self.pre_encoder(x)

        z = self.z_tokens.expand(B, -1, -1)

        # Local reasoning
        for _ in range(self.local_steps):
            z = self.reason_layer(z)

        # Global reasoning (shared concat mixing)
        gate = torch.tanh(self.global_gate)

        for _ in range(self.global_steps):
            zx = torch.cat([z, x], dim=1)
            zx = self.reason_layer(zx)
            z_new = zx[:, :self.num_z_tokens]
            z = z + gate * z_new

        # Final fusion
        x = torch.cat([z, x], dim=1)
        x = self.post_encoder(x)

        weights = torch.softmax(self.attn_pool(x), dim=1)
        x = (x * weights).sum(dim=1)

        x = self.norm(x)

        return self.classifier(x)
import torch
import torch.nn as nn
import json
import numpy as np


EXPECTED_T = 32
EXPECTED_J = 48



def uniform_sample(sequence, T=32):
    N = len(sequence)
    if N == 0:
        raise ValueError("Empty sequence")

    idxs = np.linspace(0, N - 1, T).astype(int)
    return sequence[idxs]

def compute_velocity(x):
    v = np.zeros_like(x)
    v[1:] = x[1:] - x[:-1]
    return v

def compute_acceleration(v):
    a = np.zeros_like(v)
    a[1:] = v[1:] - v[:-1]
    return a


def _sanity_check(X):
    """
    Hard fail if tensor is malformed.
    """
    if X.ndim != 3:
        raise ValueError(f"Tensor must be 3D (T,J,D), got {X.ndim}D")

    T, J, D = X.shape

    if T != EXPECTED_T:
        raise ValueError(f"Expected T={EXPECTED_T}, got {T}")

    if J != EXPECTED_J:
        raise ValueError(f"Expected J={EXPECTED_J}, got {J}")

    if not np.isfinite(X).all():
        raise ValueError("Tensor contains NaN or Inf")

    # motion must exist (non-static video)
    motion_energy = np.mean(np.abs(X[1:] - X[:-1]))
    if motion_energy < 1e-6:
        raise ValueError("Tensor has near-zero motion (static or failed extraction)")

def build_feature_tensor(kps, T=EXPECTED_T):
    """
    Build (T, J, D) tensor with automatic sanity checks.
    """
    # (T_raw, J, 2)
    if kps.ndim != 3 or kps.shape[1:] != (EXPECTED_J, 2):
        raise ValueError(f"Invalid keypoint shape: {kps.shape}")

    kps = uniform_sample(kps, T)

    vel = compute_velocity(kps)
    acc = compute_acceleration(vel)

    # Hand-relative normalization
    lh_center = kps[:, 0:21].mean(axis=1, keepdims=True)
    rh_center = kps[:, 21:42].mean(axis=1, keepdims=True)

    rel = kps.copy()
    rel[:, 0:21] -= lh_center
    rel[:, 21:42] -= rh_center

    # Concatenate: pos | vel | acc | rel
    X = np.concatenate([kps, vel, acc, rel], axis=-1).astype(np.float32)

    _sanity_check(X)
    return X


class TRMInference(nn.Module):
    def __init__(self, checkpoint,mapping_path,num_samples):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        with open(mapping_path, 'r') as f:
            self.class_map = json.load(f)
        
        # Create inverse mapping { 0: "go", 1: "happy" }
        self.idx_to_gloss = {v: k for k, v in self.class_map.items()}
        
        self.num_samples = num_samples
        self.model = TRMMicro(num_classes=len(self.class_map)).to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.valid_body_keypoints = [11, 12, 13, 14, 23, 24]  # Should match BODY_IDXS in extract_keypoints.py

    def preprocess(self, kps):
        # Implement preprocessing logic here (similar to PoseTGCNInference)
        # filter keypoints to valid body parts, kps has 21 left and 21 right hand keypoints, and rest body keypoints
        
        # Convert list to numpy array if needed
        if isinstance(kps, list):
            kps = np.array(kps, dtype=np.float32)
        
        print("🧱 Building feature tensor...")
        X = build_feature_tensor(kps, T = EXPECTED_T)          # (T, J, D)
        X = torch.from_numpy(X).float()
        if X.ndim == 3:
            X = X.view(X.shape[0], -1)          # (T, J*D)

        X = X.unsqueeze(0).to(self.device)           # (1, T, D)
        return X
    
    def predict(self, sequence_buffer):
        # 1. Preprocess the buffer
        x = self.preprocess(sequence_buffer)  # (T, D) or (T, J, D)
        # 2. Forward pass
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = logits.argmax(dim=1).item()
            confidence = probs.squeeze(0).cpu().numpy()[pred_idx]
        
        return self.idx_to_gloss[pred_idx], float(confidence)


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



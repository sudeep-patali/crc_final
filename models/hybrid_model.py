"""
hybrid_model.py
Custom hybrid CNN + Transformer architecture for histopathology patch
classification. Used for BOTH the binary (Normal/Tumor) model and the
4-class staging model -- only the final head differs.

Design rationale (for report/viva):
  - CNN branch (EfficientNet-B0, truncated): strong local inductive bias,
    captures fine-grained nuclear morphology / texture, sample-efficient.
  - Transformer branch (small ViT-style patch encoder): models long-range
    dependencies / global tissue architecture (gland arrangement, stromal
    patterning) that convolutions under-represent.
  - Cross-Attention Fusion: CNN feature tokens and Transformer tokens
    attend to each other before pooling, so the two representations are
    fused, not just concatenated -- this is what makes it a genuine hybrid
    rather than a two-model ensemble.
"""

import math
import torch
import torch.nn as nn
import torchvision.models as tv_models


# ----------------------------------------------------------------------
# CNN branch
# ----------------------------------------------------------------------
class CNNBranch(nn.Module):
    def __init__(self, pretrained=True, out_dim=256):
        super().__init__()
        backbone = tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # keep feature extractor only (drop avgpool + classifier)
        self.features = backbone.features               # output: (B, 1280, H/32, W/32)
        self.proj = nn.Conv2d(1280, out_dim, kernel_size=1)

    def forward(self, x):
        feat = self.features(x)              # (B, 1280, h, w)
        feat = self.proj(feat)                # (B, out_dim, h, w)
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)   # (B, H*W, out_dim)
        return tokens, (H, W)


# ----------------------------------------------------------------------
# Lightweight Transformer branch (patch embedding + self-attention encoder)
# ----------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=256):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                       # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)        # (B, N, embed_dim)
        return x


class TransformerEncoderBranch(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=256,
                 depth=4, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        tokens = self.patch_embed(x) + self.pos_embed
        tokens = self.encoder(tokens)
        return tokens   # (B, N, embed_dim)


# ----------------------------------------------------------------------
# Cross-Attention Fusion block
# ----------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    """CNN tokens attend to Transformer tokens and vice-versa, then fuse."""
    def __init__(self, dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.cnn_to_trans = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.trans_to_cnn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, cnn_tokens, trans_tokens):
        # CNN queries attend over Transformer keys/values (global context injected into local features)
        cnn_attended, _ = self.cnn_to_trans(cnn_tokens, trans_tokens, trans_tokens)
        cnn_out = self.norm1(cnn_tokens + cnn_attended)

        # Transformer queries attend over CNN keys/values (local detail injected into global tokens)
        trans_attended, _ = self.trans_to_cnn(trans_tokens, cnn_tokens, cnn_tokens)
        trans_out = self.norm2(trans_tokens + trans_attended)

        # global average pool both then fuse
        cnn_pooled = cnn_out.mean(dim=1)      # (B, dim)
        trans_pooled = trans_out.mean(dim=1)  # (B, dim)
        fused = self.fuse_mlp(torch.cat([cnn_pooled, trans_pooled], dim=-1))  # (B, dim)
        return fused


# ----------------------------------------------------------------------
# Full Hybrid Model
# ----------------------------------------------------------------------
class HybridCRCModel(nn.Module):
    def __init__(self, num_classes=2, embed_dim=256, img_size=224,
                 patch_size=16, transformer_depth=4, transformer_heads=4,
                 dropout=0.3, pretrained_cnn=True):
        super().__init__()
        self.cnn_branch = CNNBranch(pretrained=pretrained_cnn, out_dim=embed_dim)
        self.trans_branch = TransformerEncoderBranch(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim,
            depth=transformer_depth, num_heads=transformer_heads, dropout=dropout
        )
        self.fusion = CrossAttentionFusion(dim=embed_dim, num_heads=transformer_heads, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, x, return_features=False):
        cnn_tokens, _ = self.cnn_branch(x)
        trans_tokens = self.trans_branch(x)
        fused = self.fusion(cnn_tokens, trans_tokens)
        logits = self.classifier(fused)
        if return_features:
            return logits, fused
        return logits


def build_binary_model(pretrained=True, dropout=0.3):
    return HybridCRCModel(num_classes=2, dropout=dropout, pretrained_cnn=pretrained)


def build_stage_model(pretrained=True, dropout=0.3):
    return HybridCRCModel(num_classes=4, dropout=dropout, pretrained_cnn=pretrained)


if __name__ == "__main__":
    model = build_binary_model(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("Binary model output shape:", out.shape)  # (2, 2)

    stage_model = build_stage_model(pretrained=False)
    out2 = stage_model(dummy)
    print("Stage model output shape:", out2.shape)  # (2, 4)

"""Lazy PyTorch model builders for the saved prototype weights."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any


def require_torch():
    """Import torch only when inference code is actually used."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("cryoEMdoc inference requires torch.") from exc
    return torch


def require_torchvision_models():
    """Import torchvision models only when model construction is needed."""

    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError("cryoEMdoc inference requires torchvision.") from exc
    return models


def resolve_device(device: str = "auto"):
    """Resolve ``auto``, ``cpu``, or ``cuda`` to a torch device."""

    torch = require_torch()
    normalized = str(device).lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be one of 'auto', 'cpu', or 'cuda'")
    return torch.device(normalized)


def _torch_load(path: str | Path, device: Any, weights_only: bool = True):
    torch = require_torch()
    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=device)


def build_classifier_model(num_classes: int):
    """Build the classifier ResNet18 architecture."""

    torch = require_torch()
    nn = torch.nn
    models = require_torchvision_models()
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, int(num_classes))
    return model


def build_multitask_resnet18(num_tags: int, num_ratings: int):
    """Build the square/atlas image-only multitask ResNet18."""

    torch = require_torch()
    nn = torch.nn
    models = require_torchvision_models()

    class ResNet18Multitask(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = models.resnet18(weights=None)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.tag_head = nn.Linear(feature_dim, int(num_tags))
            self.rating_head = nn.Linear(feature_dim, int(num_ratings))

        def forward(self, x):
            features = self.backbone(x)
            return {
                "tag_logits": self.tag_head(features),
                "rating_logits": self.rating_head(features),
            }

    return ResNet18Multitask()


def build_atlas_csv_resnet18(
    num_tags: int,
    num_ratings: int,
    tabular_dim: int,
    tabular_hidden_dim: int = 32,
    tabular_dropout: float = 0.15,
):
    """Build the atlas image+CSV multitask ResNet18."""

    torch = require_torch()
    nn = torch.nn
    models = require_torchvision_models()

    class ResNet18WithTabular(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = models.resnet18(weights=None)
            image_feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.tabular_dim = int(tabular_dim)
            if self.tabular_dim > 0:
                self.tabular_branch = nn.Sequential(
                    nn.Linear(self.tabular_dim, int(tabular_hidden_dim)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=float(tabular_dropout)),
                )
                combined_dim = image_feature_dim + int(tabular_hidden_dim)
            else:
                self.tabular_branch = None
                combined_dim = image_feature_dim
            self.tag_head = nn.Linear(combined_dim, int(num_tags))
            self.rating_head = nn.Linear(combined_dim, int(num_ratings))

        def forward(self, x, tabular=None):
            image_features = self.backbone(x)
            if self.tabular_branch is not None:
                if tabular is None:
                    tabular = torch.zeros(
                        (image_features.shape[0], self.tabular_dim),
                        device=image_features.device,
                    )
                tabular_features = self.tabular_branch(tabular.float())
                features = torch.cat([image_features, tabular_features], dim=1)
            else:
                features = image_features
            return {
                "tag_logits": self.tag_head(features),
                "rating_logits": self.rating_head(features),
            }

    return ResNet18WithTabular()


def load_classifier_checkpoint(path: str | Path, device: Any, default_class_names: list[str]):
    """Load classifier checkpoint dict or plain state dict."""

    checkpoint = _torch_load(path, device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        class_names = checkpoint.get("class_names") or default_class_names
        state_dict = checkpoint["model_state_dict"]
    else:
        warnings.warn("Classifier checkpoint has no class_names; using package defaults.")
        class_names = default_class_names
        state_dict = checkpoint

    model = build_classifier_model(len(class_names)).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, list(class_names)


def load_state_dict_model(model, state_path: str | Path, device: Any):
    """Load a plain state dict into a model."""

    state = _torch_load(state_path, device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

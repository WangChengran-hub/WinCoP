"""Frozen DINOv3 backbone used by the GeoAlign-OV MVP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import timm
import torch
from safetensors.torch import load_file
from torch import Tensor, nn


DEFAULT_MODEL_NAME = "vit_base_patch16_dinov3_qkvb"
DEFAULT_WEIGHT_PATH = Path(
    "weights/dinov3/model.safetensors"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DINOv3FeatureOutput:
    """Structured output from the frozen DINOv3 backbone."""

    cls_token: Tensor
    register_tokens: Tensor
    patch_tokens: Tensor
    feature_map: Tensor
    grid_size: Tuple[int, int]


class DINOv3TimmBackbone(nn.Module):
    """Frozen timm DINOv3 ViT-B/16 feature extractor.

    Input contract:
        - Shape: [B, 3, H, W]
        - Channel order: RGB
        - Floating-point values in [0, 1]
        - H and W divisible by 16

    Output:
        - CLS token: [B, 1, 768]
        - Register tokens: [B, 4, 768]
        - Patch tokens: [B, H/16 * W/16, 768]
        - Feature map: [B, 768, H/16, W/16]
    """

    def __init__(
        self,
        weight_path: Union[str, Path] = DEFAULT_WEIGHT_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        super().__init__()

        self.weight_path = Path(weight_path)
        self.model_name = model_name

        if not self.weight_path.is_file():
            raise FileNotFoundError(
                f"DINOv3 weight file not found: {self.weight_path}"
            )

        self.model = timm.create_model(
            self.model_name,
            pretrained=False,
        )

        state_dict = load_file(
            str(self.weight_path),
            device="cpu",
        )
        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        self.model.requires_grad_(False)
        self.model.eval()

        patch_size = self.model.patch_embed.patch_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.patch_size = tuple(int(v) for v in patch_size)
        self.embed_dim = int(self.model.embed_dim)
        self.num_prefix_tokens = int(self.model.num_prefix_tokens)

        if self.patch_size != (16, 16):
            raise RuntimeError(
                f"Expected patch size (16, 16), got {self.patch_size}"
            )
        if self.embed_dim != 768:
            raise RuntimeError(
                f"Expected embedding dimension 768, got {self.embed_dim}"
            )
        if self.num_prefix_tokens != 5:
            raise RuntimeError(
                "Expected 1 CLS token and 4 register tokens, "
                f"got {self.num_prefix_tokens} prefix tokens"
            )

        mean = torch.tensor(
            IMAGENET_MEAN,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            IMAGENET_STD,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        self.register_buffer(
            "input_mean",
            mean,
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            std,
            persistent=False,
        )

        # nn.Module instances start with training=True. Explicitly place both
        # this wrapper and the wrapped timm model in permanent evaluation mode.
        self.train(False)

    def train(self, mode: bool = True) -> "DINOv3TimmBackbone":
        """Keep the frozen backbone in evaluation mode."""

        super().train(False)
        self.model.eval()
        return self

    @staticmethod
    def _validate_images(images: Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"Expected a 4D BCHW tensor, got shape {tuple(images.shape)}"
            )

        if images.shape[1] != 3:
            raise ValueError(
                f"Expected 3 RGB channels, got {images.shape[1]}"
            )

        if not images.is_floating_point():
            raise TypeError(
                f"Expected floating-point images, got dtype {images.dtype}"
            )

        height, width = images.shape[-2:]
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "Input height and width must be divisible by 16, "
                f"got {height} x {width}"
            )

        if not torch.isfinite(images).all():
            raise ValueError("Input images contain NaN or Inf values")

        min_value, max_value = torch.aminmax(images)
        if min_value.item() < 0.0 or max_value.item() > 1.0:
            raise ValueError(
                "Expected image values in [0, 1], "
                f"got range [{min_value.item():.6f}, {max_value.item():.6f}]"
            )

    @torch.inference_mode()
    def forward(self, images: Tensor) -> DINOv3FeatureOutput:
        self._validate_images(images)

        images = (
            images - self.input_mean.to(dtype=images.dtype)
        ) / self.input_std.to(dtype=images.dtype)

        tokens = self.model.forward_features(images)

        if tokens.ndim != 3:
            raise RuntimeError(
                f"Expected token tensor [B, N, C], got {tuple(tokens.shape)}"
            )

        cls_token = tokens[:, :1, :]
        register_tokens = tokens[:, 1:self.num_prefix_tokens, :]
        patch_tokens = tokens[:, self.num_prefix_tokens:, :]

        height, width = images.shape[-2:]
        grid_height = height // self.patch_size[0]
        grid_width = width // self.patch_size[1]
        expected_patches = grid_height * grid_width

        if patch_tokens.shape[1] != expected_patches:
            raise RuntimeError(
                "Patch-token count does not match the spatial grid: "
                f"tokens={patch_tokens.shape[1]}, "
                f"expected={expected_patches}"
            )

        feature_map = (
            patch_tokens
            .transpose(1, 2)
            .reshape(
                images.shape[0],
                self.embed_dim,
                grid_height,
                grid_width,
            )
            .contiguous()
        )

        return DINOv3FeatureOutput(
            cls_token=cls_token,
            register_tokens=register_tokens,
            patch_tokens=patch_tokens,
            feature_map=feature_map,
            grid_size=(grid_height, grid_width),
        )

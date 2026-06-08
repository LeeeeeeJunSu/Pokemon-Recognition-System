from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.vision_transformer import (
    VisionTransformer,
    interpolate_embeddings,
)


def build_vit_b_16(
    image_size: int,
    patch_size: int,
    use_pretrained: bool,
    pretrained_weights: str,
) -> VisionTransformer:
    if image_size <= 0 or patch_size <= 0:
        raise ValueError("image_size and patch_size must be positive.")
    if image_size % patch_size != 0:
        raise ValueError(
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})."
        )

    weights = None
    if use_pretrained:
        try:
            weights = ViT_B_16_Weights[pretrained_weights]
        except KeyError as error:
            raise ValueError(
                f"Unsupported pretrained weights: {pretrained_weights}"
            ) from error

    if image_size == 224 and patch_size == 16:
        return vit_b_16(weights=weights)

    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
    )
    if weights is None:
        return model

    state_dict = OrderedDict(
        weights.get_state_dict(progress=True, check_hash=True)
    )
    patch_weights = state_dict["conv_proj.weight"]
    if patch_weights.shape[-2:] != (patch_size, patch_size):
        output_channels, input_channels, _, _ = patch_weights.shape
        resized = F.interpolate(
            patch_weights.reshape(
                output_channels * input_channels,
                1,
                patch_weights.shape[-2],
                patch_weights.shape[-1],
            ),
            size=(patch_size, patch_size),
            mode="bicubic",
            align_corners=True,
        )
        state_dict["conv_proj.weight"] = resized.reshape(
            output_channels,
            input_channels,
            patch_size,
            patch_size,
        )

    state_dict = interpolate_embeddings(
        image_size=image_size,
        patch_size=patch_size,
        model_state=state_dict,
        interpolation_mode="bicubic",
        reset_heads=True,
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("heads.")]
    missing = [key for key in incompatible.missing_keys if not key.startswith("heads.")]
    if unexpected or missing:
        raise RuntimeError(
            f"Could not adapt pretrained ViT weights. Missing={missing}, unexpected={unexpected}"
        )
    return model
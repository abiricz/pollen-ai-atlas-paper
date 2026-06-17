# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import timm

def tile_image(image, tile_size):
    """
    Tiles the input image into non-overlapping patches of size tile_size x tile_size.
    Args:
        image: Input tensor of shape (B, C, H, W).
        tile_size: Size of each tile.
    Returns:
        Tiled patches as a tensor of shape (B * num_tiles, C, tile_size, tile_size).
    """
    B, C, H, W = image.shape
    assert H % tile_size == 0 and W % tile_size == 0, "Image dimensions must be divisible by tile size."
    
    num_tiles_h = H // tile_size
    num_tiles_w = W // tile_size
    tiles = image.unfold(2, tile_size, tile_size).unfold(3, tile_size, tile_size)
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, tile_size, tile_size)
    return tiles, num_tiles_h, num_tiles_w


# Utility: Reconstruct Feature Map
def reconstruct_feature_map(tiles, num_tiles_h, num_tiles_w):
    """
    Reconstructs a unified feature map from tiled feature maps.
    Args:
        tiles: Tiled feature maps of shape (B * num_tiles, C, H, W).
        num_tiles_h: Number of tiles along the height.
        num_tiles_w: Number of tiles along the width.
    Returns:
        Reconstructed feature map of shape (B, C, H * num_tiles_h, W * num_tiles_w).
    """
    B_tiles, C, H, W = tiles.shape
    B = B_tiles // (num_tiles_h * num_tiles_w)
    tiles = tiles.view(B, num_tiles_h, num_tiles_w, C, H, W)
    return tiles.permute(0, 3, 1, 4, 2, 5).reshape(B, C, num_tiles_h * H, num_tiles_w * W)

class ViTBackbone(nn.Module):
    def __init__(self, backbone_model, 
                model_img_input_size, conv_reduce=False, output_channels=256):
        super().__init__()
        
        self.vit = backbone_model
        self.conv_reduce = conv_reduce
        self.model_img_input_size = model_img_input_size
        
        # Freeze all parameters in the ViT model
        for param in self.vit.parameters():
            param.requires_grad = False

        # 1x1 convolution to adjust output channels to the desired size
        if self.conv_reduce:
            self.feature_adjust = nn.Conv2d(1024, output_channels, kernel_size=1) # HERE 1024 is HARDCODED !

    def _process_single_tile(self, x):
        """
        Process a single tile or small input directly through the ViT model.
        """
        # Patchify and add positional embeddings
        patch_embed = self.vit.patch_embed(x)  # Patchify image
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # Expand CLS token
        pos_embed = self.vit.pos_embed  # Positional embeddings

        x = torch.cat((cls_token, patch_embed), dim=1)  # Concatenate CLS token
        x = x + pos_embed  # Add positional embeddings

        # Forward pass through transformer blocks
        for block in self.vit.blocks:
            x = block(x)

        # Normalize and extract patch tokens
        x = self.vit.norm(x)
        patch_tokens = x[:, 1:, :]  # Exclude CLS token

        # Reshape patch tokens into spatial grid
        B, T, C = patch_tokens.shape
        grid_size = int(T ** 0.5)
        spatial_tokens = patch_tokens.view(B, grid_size, grid_size, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        return spatial_tokens

    
    def forward(self, x):
        """
        Processes an image or tiled images through the ViT backbone.
        Automatically handles tiling and reconstruction for larger inputs.
        """
        # Check if tiling is needed
        B, C, H, W = x.shape
        if H % self.model_img_input_size != 0 or W % self.model_img_input_size != 0:
            raise ValueError(f"Input height and width must be multiples of {self.model_img_input_size}.")
        
        if H > self.model_img_input_size or W > self.model_img_input_size:
            # Tile the input
            tiles, num_tiles_h, num_tiles_w = tile_image(x, tile_size=self.model_img_input_size) # note: this works with squares for now !

            # Process each tile through the ViT backbone
            features = []
            for tile in tiles:
                tile = tile.unsqueeze(0)  # Add batch dimension
                features.append(self._process_single_tile(tile))

            # Concatenate and reconstruct the feature map
            features = torch.cat(features, dim=0)
            unified_features = reconstruct_feature_map(features, num_tiles_h, num_tiles_w)
        else:
            # Process single input directly
            unified_features = self._process_single_tile(x)
            
        if self.conv_reduce:
            self.feature_adjust(unified_features)  # Apply 1x1 convolution
        else:
            return unified_features # Return ViT output as is

class NamedViTBackbone(nn.Module):
    def __init__(self, 
                backbone_model,
                model_img_input_size,
                conv_reduce=False,
                output_channels=256):
        super().__init__()
        self.body = ViTBackbone(backbone_model, model_img_input_size, conv_reduce, output_channels) # Add the ViT backbone as "body"
        self.out_channels = output_channels  # Specify the output channels

    def forward(self, x):
        # Forward pass through the backbone
        features = self.body(x)
        if isinstance(features, torch.Tensor):  # Ensure features is a tensor
            features = {"0": features}  # Wrap it into a dictionary
        return features

class FixedSizeTransform(GeneralizedRCNNTransform):
    def __init__(self, tile_size, image_mean, image_std):
        # Use min_size=max_size=518 to prevent resizing
        super().__init__(min_size=tile_size, max_size=tile_size, image_mean=image_mean, image_std=image_std)

    def resize(self, image, target):
        # Override to bypass resizing
        return image, target

    def batch_images(self, images, size_divisible=32):
        # Override this to stop padding to multiples of 32
        max_size = tuple(max(s) for s in zip(*[img.shape[-2:] for img in images]))
        batch_shape = (len(images), 3, max_size[0], max_size[1])  # Do not pad to 32-multiples
        batched_imgs = images[0].new_full(batch_shape, 0)  # Keep original size
        for img, pad_img in zip(images, batched_imgs):
            pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
        return batched_imgs
    
def model_selection(config):
    """
    Select and return a pre-trained ViT model based on the input name or index.

    Parameters:
        config (str | int):
            If str, the name of the model to load.
            If int, the index of the model in the predefined list.

    Returns:
        timm.models.VisionTransformer: A Vision Transformer model.
    """
    # List of all possible model names
    model_names_vit = [
        "vit_large_patch16_224",
        "vit_small_patch14_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",
        "vit_large_patch14_dinov2.lvd142m"
    ]

    if isinstance(config, int):
        if config < 0 or config >= len(model_names_vit):
            raise ValueError(f"Index out of range. Valid indices: 0 to {len(model_names_vit) - 1}.")
        model_name = model_names_vit[config]
    elif isinstance(config, str):
        if config not in model_names_vit:
            raise ValueError(f"Model name not recognized. Choose from: {model_names_vit}.")
        model_name = config
    else:
        raise TypeError("Config must be a str or int.")

    # Load the model
    vit_backbone = timm.create_model(
        model_name,
        pretrained=True if "patch14" in model_name else False, # True,
        img_size=518 if "patch14" in model_name else 224,
        init_values=1e-5,
        num_classes=0  # Remove classifier nn.Linear
    )

    # Load pre-trained weights for specific models, if necessary
    if model_name == "vit_large_patch16_224":
        print('Loading uni weights..')
        vit_backbone.load_state_dict(
            torch.load("../training/vit_large_patch16_224.dinov2_uni_mass100k.pth", map_location="cpu"), strict=False
        )
    model_img_input_size = 518 if "patch14" in model_name else 224
    
    return vit_backbone, model_img_input_size

# ADDED
class CONCHBackbone(nn.Module):
    def __init__(self, conch_model, patch_size=16):
        super().__init__()
        self.trunk = conch_model.visual.trunk  # VisionTransformer
        self.patch_size = patch_size

        # Freeze if needed
        for p in self.trunk.parameters():
            p.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape

        # Step 1: patchify
        x = self.trunk.patch_embed(x)  # Shape: (B, embed_dim, H', W')

        # Step 2: flatten spatial → sequence
        x = x.flatten(2).transpose(1, 2)  # Shape: (B, num_patches, D)

        # Step 3: add positional embeddings + CLS
        cls_token = self.trunk.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_token, x), dim=1)

        x = x + self.trunk.pos_embed  # positional encodings
        x = self.trunk.pos_drop(x)

        # Step 4: transformer blocks
        for blk in self.trunk.blocks:
            x = blk(x)

        # Step 5: final layer norm
        x = self.trunk.norm(x)

        # Step 6: return full tokens, not just CLS
        return x[:, 1:, :]  # Exclude CLS → (B, N, D)
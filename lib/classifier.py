# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

"""
Pollen Classifier Module
========================

Provides the PollenClassifier class for filtering mining candidates.
Uses the finetuned ViT-Small-LVD backbone with a classification head.

The classifier distinguishes:
- 51 pollen species/genera classes (index 0-50)
- 1 background class (index 51)

Usage:
    from lib.classifier import PollenClassifier
    
    classifier = PollenClassifier(
        checkpoint_path="01_initialization/weights_vit_small_lvd_*.pth",
        device="cuda:0"
    )
    
    # Single image
    class_id, class_name, prob = classifier.predict(image)
    
    # Batch processing
    results = classifier.predict_batch(images)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
from typing import Tuple, List, Dict, Optional, Union

# Class mapping from finetuning notebook
# Index 0-50: 51 pollen species, Index 51: background
# Extracted from: uqs_species output in Tif_clustering_embedder_finetuning.ipynb
CLASSES_TO_INT = {
    'acer': 0,
    'alnus': 1,
    'ambrosia': 2,
    'betula': 3,
    'brassica_napus': 4,
    'brassicaceae': 5,
    'cannabis': 6,
    'carpinus': 7,
    'causarina': 8,
    'cedrus': 9,
    'cheno': 10,
    'corylus': 11,
    'cupr': 12,
    'cupressus': 13,
    'ericaceae': 14,
    'fabaceae': 15,
    'festuca': 16,
    'forsythia': 17,
    'fraxinus': 18,
    'ginkgo': 19,
    'hedera': 20,
    'humulus_japonicus': 21,
    'hun_betula': 22,
    'hun_corylus': 23,
    'iva': 24,
    'juglans': 25,
    'juncaceae': 26,
    'ligustrum': 27,
    'ligustrum_2': 28,
    'mimosa': 29,
    'morus_alba': 30,
    'olea': 31,
    'palmaceae': 32,
    'parietaria': 33,
    'phacelia': 34,
    'picea': 35,
    'pinus': 36,
    'plantago': 37,
    'platanus': 38,
    'poaceae': 39,
    'quercus': 40,
    'ranunculus': 41,
    'rumex': 42,
    'salix': 43,
    'tilia': 44,
    'triticum_aestivum': 45,
    'typha': 46,
    'ulmus': 47,
    'urti': 48,
    'urtica': 49,
    'vitis': 50,
    'background': 51,  # Background/non-pollen class
}

# Reverse mapping: int -> class name
INT_TO_CLASSES = {v: k for k, v in CLASSES_TO_INT.items()}

# Number of classes (including background)
NUM_CLASSES = len(CLASSES_TO_INT)  # 52 total

# Background class index
BACKGROUND_CLASS_ID = CLASSES_TO_INT['background']


def get_val_transform(img_size: int = 518) -> transforms.Compose:
    """Standard validation transform matching training."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])


class PollenClassifier(nn.Module):
    """
    Pollen grain classifier using finetuned ViT-Small-LVD.
    
    The model predicts one of 52 classes:
    - 51 pollen species/genera (indices 0-50)  
    - 1 background class (index 51)
    
    For filtering, candidates where argmax == background are rejected.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        img_size: int = 518,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        
        self.device = torch.device(device)
        self.img_size = img_size
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        
        # Build model
        self.model = self._build_model(model_name, num_classes)
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Move to device and set eval mode
        self.model = self.model.to(self.device).eval()
        
        # Transforms
        self.transform = get_val_transform(img_size)
        
        print(f"[Classifier] Loaded from {checkpoint_path}")
        print(f"[Classifier] {num_classes} classes, background_id={BACKGROUND_CLASS_ID}")
    
    def _build_model(self, model_name: str, num_classes: int) -> nn.Module:
        """Build the ViT model with classification head."""
        
        # Determine embedding dimension
        if 'small' in model_name:
            in_features = 384
        elif 'base' in model_name:
            in_features = 768
        elif 'large' in model_name:
            in_features = 1024
        else:
            in_features = 384  # default
        
        # Create model without classifier
        model = timm.create_model(
            model_name,
            pretrained=False,  # Will load from checkpoint
            img_size=self.img_size,
            init_values=1e-5,
            num_classes=0,  # Remove default head
        )
        
        # Add classification head (matches finetuning notebook)
        model.head = nn.Linear(in_features, num_classes)
        
        return model
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint weights."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Load with strict=False to handle potential mismatches
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"[Classifier] Warning: Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"[Classifier] Warning: Unexpected keys: {unexpected[:5]}...")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.model(x)
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> Tuple[int, str, float]:
        """
        Predict class for a single image.
        
        Args:
            image: PIL Image, numpy array (H,W,3), or tensor (3,H,W)
            
        Returns:
            class_id: Predicted class index
            class_name: Human-readable class name
            probability: Softmax probability for predicted class
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Transform and add batch dim
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        
        # Get prediction
        prob, class_id = probs.max(dim=1)
        class_id = class_id.item()
        prob = prob.item()
        class_name = INT_TO_CLASSES.get(class_id, f"class_{class_id}")
        
        return class_id, class_name, prob
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Predict classes for a batch of images.
        
        Args:
            images: List of PIL Images or numpy arrays
            batch_size: Processing batch size
            
        Returns:
            List of dicts with keys: class_id, class_name, probability, is_pollen
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Convert and stack
            tensors = []
            for img in batch_images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img.astype(np.uint8))
                tensors.append(self.transform(img))
            
            x = torch.stack(tensors).to(self.device)
            
            # Forward pass
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            
            # Get predictions
            max_probs, class_ids = probs.max(dim=1)
            
            for j in range(len(batch_images)):
                cid = class_ids[j].item()
                prob = max_probs[j].item()
                cname = INT_TO_CLASSES.get(cid, f"class_{cid}")
                
                results.append({
                    'class_id': cid,
                    'class_name': cname,
                    'probability': prob,
                    'is_pollen': cid != BACKGROUND_CLASS_ID,
                })
        
        return results
    
    @torch.no_grad()
    def get_all_probs(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get probabilities for all classes.
        
        Returns:
            probs: Array of shape (num_classes,) with probabilities
            class_names: List of class names in order
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        class_names = [INT_TO_CLASSES.get(i, f"class_{i}") for i in range(self.num_classes)]
        
        return probs, class_names
    
    def is_pollen(self, class_id: int) -> bool:
        """Check if class_id represents a pollen grain (not background)."""
        return class_id != BACKGROUND_CLASS_ID
    
    def filter_candidates(
        self,
        class_ids: np.ndarray,
        probabilities: np.ndarray,
        prob_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Return boolean mask of candidates to keep.
        
        Keeps candidates where:
        - class_id != background AND
        - probability >= threshold
        
        Args:
            class_ids: Array of predicted class indices
            probabilities: Array of class probabilities
            prob_threshold: Minimum probability to keep
            
        Returns:
            Boolean mask array (True = keep)
        """
        is_pollen = class_ids != BACKGROUND_CLASS_ID
        above_threshold = probabilities >= prob_threshold
        return is_pollen & above_threshold


def load_classifier(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda:0",
) -> PollenClassifier:
    """
    Convenience function to load the classifier.
    
    If checkpoint_path is None, uses default production checkpoint.
    """
    if checkpoint_path is None:
        # Default to production checkpoint
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(
            project_root,
            "01_initialization",
            "weights_vit_small_lvd_20250620_0312.pth"
        )
    
    return PollenClassifier(checkpoint_path=checkpoint_path, device=device)


# Export
__all__ = [
    'PollenClassifier',
    'load_classifier',
    'CLASSES_TO_INT',
    'INT_TO_CLASSES',
    'NUM_CLASSES',
    'BACKGROUND_CLASS_ID',
    'get_val_transform',
]

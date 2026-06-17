# Copyright 2026 András Biricz. Licensed under the Apache License, Version 2.0.

"""
Species Mapping Module
======================
Loads species mappings directly from HITL-curated caption anchor files.

NO HARDCODING - everything is loaded from:
  03_captioning/caption_anchors/*_species.txt

The filename (without _species.txt) is the slide name.
The file content is the validated species name.

Example:
  acer_edf_species.txt contains "Alnus" 
    → slide "acer_edf" is actually Alnus, not Acer!
"""

import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple


def load_caption_anchors(caption_anchors_dir: Optional[Path] = None) -> Dict[str, str]:
    """
    Load slide → species mapping from caption anchor files.
    
    Args:
        caption_anchors_dir: Path to caption_anchors folder.
            Defaults to 03_captioning/caption_anchors relative to repo root.
    
    Returns:
        Dictionary mapping slide name → HITL-curated species name
    """
    if caption_anchors_dir is None:
        # Find repo root
        current = Path(__file__).parent.parent
        caption_anchors_dir = current / "03_captioning" / "caption_anchors"
    
    caption_anchors_dir = Path(caption_anchors_dir)
    
    if not caption_anchors_dir.exists():
        raise FileNotFoundError(
            f"Caption anchors directory not found: {caption_anchors_dir}"
        )
    
    mapping: Dict[str, str] = {}
    
    for species_file in caption_anchors_dir.glob("*_species.txt"):
        slide_name = species_file.stem.replace("_species", "")
        species = species_file.read_text().strip()
        mapping[slide_name] = species
    
    return mapping


def build_class_mappings(
    caption_anchors: Optional[Dict[str, str]] = None
) -> Tuple[Dict[str, int], Dict[int, str], int]:
    """
    Build class mappings from caption anchors.
    
    Returns:
        (species_to_int, int_to_species, num_classes)
        
    Species are sorted alphabetically, "Unknown" is last class.
    """
    if caption_anchors is None:
        caption_anchors = load_caption_anchors()
    
    # Get unique species, sorted
    unique_species = sorted(set(caption_anchors.values()))
    
    # Remove "Unknown" from sorted list if present (will add at end)
    unique_species = [s for s in unique_species if s.lower() != "unknown"]
    
    # Build mappings
    species_to_int: Dict[str, int] = {}
    int_to_species: Dict[int, str] = {}
    
    for i, species in enumerate(unique_species):
        species_to_int[species.lower()] = i
        int_to_species[i] = species
    
    # Add Unknown/background as last class
    background_id = len(unique_species)
    species_to_int["unknown"] = background_id
    species_to_int["background"] = background_id
    int_to_species[background_id] = "Unknown"
    
    num_classes = len(unique_species) + 1
    
    return species_to_int, int_to_species, num_classes


def get_slide_class_id(
    slide_name: str,
    caption_anchors: Dict[str, str],
    species_to_int: Dict[str, int]
) -> int:
    """
    Get class ID for a slide using caption anchors.
    
    Args:
        slide_name: Name of the slide
        caption_anchors: slide → species mapping
        species_to_int: species → class_id mapping
    
    Returns:
        Integer class ID
    """
    species = caption_anchors.get(slide_name, "Unknown")
    return species_to_int.get(species.lower(), species_to_int.get("unknown", 0))


def get_class_names(caption_anchors: Optional[Dict[str, str]] = None) -> List[str]:
    """Get list of all class names in order."""
    _, int_to_species, num_classes = build_class_mappings(caption_anchors)
    return [int_to_species[i] for i in range(num_classes)]


def get_trainable_slides(caption_anchors: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, str], List[str]]:
    """
    Get slides with valid species labels (excluding Unknown).
    
    Returns:
        (trainable_anchors, excluded_slides)
        
    Use this to exclude slides labeled as "Unknown" from training and evaluation.
    """
    if caption_anchors is None:
        caption_anchors = load_caption_anchors()
    
    trainable = {}
    excluded = []
    
    for slide, species in caption_anchors.items():
        if species.lower() == "unknown":
            excluded.append(slide)
        else:
            trainable[slide] = species
    
    return trainable, excluded


if __name__ == "__main__":
    print("=== Species Mapping Validation ===\n")
    
    # Load from files
    anchors = load_caption_anchors()
    species_to_int, int_to_species, num_classes = build_class_mappings(anchors)
    
    print(f"Loaded {len(anchors)} slides from caption_anchors/")
    print(f"Unique species: {num_classes - 1} + 1 Unknown = {num_classes} classes")
    
    # Show examples
    print("\n=== Example Mappings ===")
    examples = ["acer_edf", "mimosa_edf", "hun_betula_edf"]
    for slide in examples:
        if slide in anchors:
            species = anchors[slide]
            class_id = species_to_int.get(species.lower(), -1)
            print(f"  {slide} → {species} (class_id={class_id})")
    
    # Show all classes
    print("\n=== All Classes ===")
    for i in range(num_classes):
        print(f"  {i:2d}: {int_to_species[i]}")

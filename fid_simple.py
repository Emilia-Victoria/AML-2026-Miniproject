"""
Simple FID Evaluation using pytorch-fid library

This module provides a streamlined way to compute FID scores using the 
pytorch-fid library, which handles InceptionV3 features and FID calculation.

Installation:
    pip install pytorch-fid

Usage:
    from fid_simple import compute_fid_from_checkpoint
    fid_score = compute_fid_from_checkpoint(
        checkpoint_path='path/to/checkpoint.pth',
        generator_class=MyGenerator,
        real_images_dir='path/to/real/images',
        num_samples=1000,
        device='cuda'
    )
"""

import torch
import torch.nn as nn
from pytorch_fid.fid_score import calculate_fid_given_paths
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from typing import Callable, Optional


def generate_images_to_directory(
    generator: nn.Module,
    output_dir: str,
    num_images: int,
    latent_dim: int,
    batch_size: int = 32,
    device: str = 'cuda'
) -> None:
    """
    Generate images from a generator and save them to a directory.
    
    Args:
        generator: Generator model in eval mode
        output_dir: Directory to save generated images
        num_images: Number of images to generate
        latent_dim: Dimension of latent space
        batch_size: Batch size for generation
        device: Device to run on
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator.eval()
    num_batches = (num_images + batch_size - 1) // batch_size
    
    image_count = 0
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating images", leave=False):
            current_batch_size = min(batch_size, num_images - image_count)
            
            noise = torch.randn(current_batch_size, latent_dim).to(device)
            images = generator(noise)
            
            # Rescale from [-1, 1] to [0, 1]
            images = (images + 1) / 2
            images = torch.clamp(images, 0, 1)
            
            # Save images
            for i in range(images.shape[0]):
                img = images[i].cpu()
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                
                img_pil = Image.fromarray(img)
                img_path = os.path.join(output_dir, f"generated_{image_count:05d}.png")
                img_pil.save(img_path)
                
                image_count += 1
                if image_count >= num_images:
                    break


def compute_fid_from_checkpoint(
    checkpoint_path: str,
    generator_class: Callable,
    real_images_dir: str,
    num_samples: int = 1000,
    latent_dim: int = 100,
    device: str = 'cuda',
    batch_size: int = 32,
    temp_dir: Optional[str] = None
) -> float:
    """
    Compute FID score for a generator checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        generator_class: Generator class to instantiate
        real_images_dir: Directory containing real images
        num_samples: Number of samples to use for FID
        latent_dim: Latent dimension
        device: Device to run on
        batch_size: Batch size for generation
        temp_dir: Temporary directory for generated images (default: /tmp/fid_gen)
    
    Returns:
        fid_score: FID score (lower is better)
    """
    if temp_dir is None:
        temp_dir = '/tmp/fid_gen' if os.name != 'nt' else './fid_gen'
    
    # Load generator
    print(f"Loading checkpoint: {checkpoint_path}")
    generator = generator_class(latent_dim=latent_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    # Generate images
    print(f"Generating {num_samples} images...")
    generate_images_to_directory(
        generator, temp_dir, num_samples, latent_dim, batch_size, device
    )
    
    # Compute FID
    print(f"Computing FID score...")
    fid_score = calculate_fid_given_paths(
        [real_images_dir, temp_dir],
        batch_size=batch_size,
        device=device,
        dims=2048
    )
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
    return fid_score


def evaluate_all_checkpoints_simple(
    checkpoint_dirs: dict,
    generator_classes: dict,
    real_images_dir: str,
    latent_dim: int = 100,
    num_samples: int = 1000,
    device: str = 'cuda'
) -> dict:
    """
    Evaluate all checkpoints using pytorch-fid.
    
    Args:
        checkpoint_dirs: Dict mapping model names to checkpoint directories
        generator_classes: Dict mapping model names to generator classes
        real_images_dir: Directory with real images
        latent_dim: Latent dimension
        num_samples: Samples for FID calculation
        device: Device to run on
    
    Returns:
        results: Dict with FID scores
    """
    results = {}
    
    for model_name, checkpoint_dir in checkpoint_dirs.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}")
        print(f"{'='*60}")
        
        generator_class = generator_classes[model_name]
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
        
        model_results = {}
        for checkpoint_file in checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            print(f"\n{checkpoint_file}")
            
            try:
                fid_score = compute_fid_from_checkpoint(
                    checkpoint_path,
                    generator_class,
                    real_images_dir,
                    num_samples,
                    latent_dim,
                    device
                )
                model_results[checkpoint_file] = fid_score
                print(f"✓ FID: {fid_score:.4f}")
            except Exception as e:
                print(f"✗ Error: {e}")
                model_results[checkpoint_file] = None
        
        results[model_name] = model_results
    
    # Print summary
    print("\n" + "="*60)
    print("FID Evaluation Summary (lower is better)")
    print("="*60)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        valid = [(k, v) for k, v in model_results.items() if v is not None]
        if valid:
            valid.sort(key=lambda x: x[1])
            for checkpoint_file, fid_score in valid:
                print(f"  {checkpoint_file:40s} FID: {fid_score:.4f}")
    
    return results

"""
FID (Fréchet Inception Distance) Evaluation for GAN Models

This module provides utilities to compute FID scores for generative models,
allowing comparison of models trained with different hyperparameters.

FID is a metric that measures the distance between the distribution of
generated images and real images using features from a pre-trained
InceptionV3 model.

Lower FID scores indicate better quality and diversity of generated images.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from glob import glob
from typing import Tuple, Dict, List


def get_inception_features(images: torch.Tensor, inception_model: nn.Module, device: str) -> np.ndarray:
    """
    Extract features from images using pre-trained InceptionV3 model.
    
    Args:
        images: Tensor of shape [N, 3, H, W] with values in [0, 1]
        inception_model: Pre-trained InceptionV3 model
        device: Device to run on
    
    Returns:
        features: Numpy array of shape [N, 2048] containing the feature vectors
    """
    with torch.no_grad():
        # InceptionV3 expects input in range [0, 1]
        # Get the features from the final avg pool layer
        features = inception_model(images)
    
    return features.cpu().numpy()


def compute_fid(
    real_images_loader: DataLoader,
    generator: nn.Module,
    num_samples: int,
    latent_dim: int,
    device: str,
    batch_size: int = 64
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Fréchet Inception Distance between real and generated images.
    
    The FID score is computed as:
    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r * Σ_g)^(1/2))
    
    where μ and Σ are the mean and covariance of the feature distributions.
    
    Args:
        real_images_loader: DataLoader with real images (values in [-1, 1])
        generator: Generator model in eval mode
        num_samples: Number of samples to use for FID computation
        latent_dim: Dimension of latent space
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for feature extraction
    
    Returns:
        Tuple containing:
            - fid_score: FID score (lower is better)
            - real_mean: Mean of real image features
            - real_cov: Covariance of real image features
            - fake_mean: Mean of generated image features
            - fake_cov: Covariance of generated image features
    """
    
    # Load pre-trained InceptionV3 model
    print("Loading InceptionV3 model...")
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.to(device)
    inception_model.eval()
    
    # Modify the model to return features instead of classification output
    inception_model.fc = nn.Identity()
    
    # Extract real image features
    real_features = []
    generator.eval()
    
    print("Extracting real image features...")
    for real_images in tqdm(real_images_loader, desc="Real images", leave=False):
        real_images = real_images.to(device)
        
        # Rescale from [-1, 1] to [0, 1]
        real_images = (real_images + 1) / 2
        real_images = torch.clamp(real_images, 0, 1)
        
        # Resize to 299x299 (InceptionV3 input size)
        real_images_resized = torch.nn.functional.interpolate(
            real_images, size=(299, 299), mode='bilinear', align_corners=False
        )
        
        features = get_inception_features(real_images_resized, inception_model, device)
        real_features.append(features)
        
        if len(np.concatenate(real_features)) >= num_samples:
            break
    
    real_features = np.concatenate(real_features)[:num_samples]
    print(f"Extracted {len(real_features)} real image features")
    
    # Extract generated image features
    fake_features = []
    
    print("Extracting generated image features...")
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generated images", leave=False):
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(noise)
            
            # Rescale from [-1, 1] to [0, 1]
            fake_images = (fake_images + 1) / 2
            fake_images = torch.clamp(fake_images, 0, 1)
            
            # Resize to 299x299
            fake_images_resized = torch.nn.functional.interpolate(
                fake_images, size=(299, 299), mode='bilinear', align_corners=False
            )
            
            features = get_inception_features(fake_images_resized, inception_model, device)
            fake_features.append(features)
    
    fake_features = np.concatenate(fake_features)[:num_samples]
    print(f"Extracted {len(fake_features)} generated image features")
    
    # Compute mean and covariance
    real_mean = np.mean(real_features, axis=0)
    real_cov = np.cov(real_features.T)
    
    fake_mean = np.mean(fake_features, axis=0)
    fake_cov = np.cov(fake_features.T)
    
    # Compute FID
    diff = real_mean - fake_mean
    
    # Compute the product inside the square root
    # Handle potential numerical issues with complex numbers
    cov_product = real_cov @ fake_cov
    
    try:
        cov_sqrt = sqrtm(cov_product)
        
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
    except Exception as e:
        print(f"Warning: Error computing matrix square root: {e}")
        cov_sqrt = np.zeros_like(cov_product)
    
    fid_score = np.sum(diff ** 2) + np.trace(real_cov + fake_cov - 2 * cov_sqrt)
    
    return fid_score, real_mean, real_cov, fake_mean, fake_cov


def evaluate_checkpoint(
    checkpoint_path: str,
    generator_class,
    image_paths: List[str],
    latent_dim: int,
    num_samples: int = 1000,
    device: str = "cpu",
    batch_size: int = 32
) -> float:
    """
    Evaluate a single checkpoint and return its FID score.
    
    Args:
        checkpoint_path: Path to checkpoint file
        generator_class: Generator class to instantiate
        image_paths: List of paths to real images
        latent_dim: Dimension of latent space
        num_samples: Number of samples for FID computation
        device: Device to run on
        batch_size: Batch size for data loading
    
    Returns:
        fid_score: FID score for the checkpoint
    """
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms
    
    # Simple dataset class
    class SimpleImageDataset(Dataset):
        def __init__(self, image_paths, transform=None):
            self.image_paths = image_paths
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx])
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            else:
                img = img.convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            else:
                # Default transform
                img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
                img = torch.from_numpy(img_array).permute(2, 0, 1)
            
            return img
    
    # Create dataset and loader
    dataset = SimpleImageDataset(image_paths[:num_samples])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Load generator
    generator = generator_class().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['state_dict'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    
    # Compute FID
    fid_score, _, _, _, _ = compute_fid(
        loader, generator, num_samples, latent_dim, device
    )
    
    return fid_score


def evaluate_all_checkpoints(
    checkpoint_dirs: Dict[str, str],
    generator_classes: Dict[str, callable],
    image_paths: List[str],
    latent_dim: int,
    num_samples: int = 1000,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluate all checkpoints in multiple directories and return FID scores.
    
    Args:
        checkpoint_dirs: Dictionary mapping model names to checkpoint directories
        generator_classes: Dictionary mapping model names to generator classes
        image_paths: List of paths to real images
        latent_dim: Dimension of latent space
        num_samples: Number of samples for FID computation
        device: Device to run on
    
    Returns:
        results: Dictionary with model/checkpoint info and FID scores
    """
    results = {}
    
    for model_name, checkpoint_dir in checkpoint_dirs.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} checkpoints from: {checkpoint_dir}")
        print(f"{'='*60}")
        
        if model_name not in generator_classes:
            print(f"Error: Generator class for {model_name} not provided")
            continue
        
        generator_class = generator_classes[model_name]
        checkpoints = sorted(glob(os.path.join(checkpoint_dir, "*.pth")))
        
        if not checkpoints:
            print(f"No checkpoints found in {checkpoint_dir}")
            continue
        
        model_results = {}
        for checkpoint_path in checkpoints:
            checkpoint_name = os.path.basename(checkpoint_path)
            print(f"\nEvaluating: {checkpoint_name}")
            
            try:
                fid_score = evaluate_checkpoint(
                    checkpoint_path,
                    generator_class,
                    image_paths,
                    latent_dim,
                    num_samples,
                    device
                )
                model_results[checkpoint_name] = fid_score
                print(f"✓ FID Score: {fid_score:.4f}")
            except Exception as e:
                print(f"✗ Error evaluating {checkpoint_name}: {e}")
                model_results[checkpoint_name] = None
        
        results[model_name] = model_results
    
    # Print summary
    print("\n" + "="*60)
    print("FID Evaluation Summary (lower is better)")
    print("="*60)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        valid_results = {k: v for k, v in model_results.items() if v is not None}
        if valid_results:
            sorted_results = sorted(valid_results.items(), key=lambda x: x[1])
            for checkpoint_name, fid_score in sorted_results:
                print(f"  {checkpoint_name:40s} FID: {fid_score:.4f}")
        else:
            print("  No valid results for this model")
    
    return results


def compare_hyperparameters(
    experiment_results: Dict[str, Dict[str, float]],
    metric_name: str = "FID"
) -> None:
    """
    Create a summary comparing FID scores across different hyperparameter configurations.
    
    Args:
        experiment_results: Dictionary with experiment names and their FID scores
        metric_name: Name of the metric being compared
    """
    print("\n" + "="*70)
    print(f"{metric_name} Comparison Across Hyperparameters")
    print("="*70)
    
    for exp_name, scores in experiment_results.items():
        if isinstance(scores, dict):
            valid_scores = [v for v in scores.values() if v is not None]
            if valid_scores:
                best = min(valid_scores)
                worst = max(valid_scores)
                avg = np.mean(valid_scores)
                print(f"\n{exp_name}:")
                print(f"  Best  {metric_name}: {best:.4f}")
                print(f"  Worst {metric_name}: {worst:.4f}")
                print(f"  Avg   {metric_name}: {avg:.4f}")
        else:
            print(f"\n{exp_name}: {metric_name} = {scores:.4f}")

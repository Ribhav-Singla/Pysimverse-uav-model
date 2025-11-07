"""
Depth Image Preprocessing Pipeline
Provides comprehensive preprocessing for depth images before CNN processing
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available, using basic preprocessing")


class DepthPreprocessor:
    """
    Comprehensive preprocessing pipeline for depth images
    Handles normalization, noise reduction, and enhancement
    """
    
    def __init__(self, 
                 image_size: int = 64,
                 depth_range: Tuple[float, float] = (0.1, 10.0),
                 enable_denoising: bool = True,
                 enable_enhancement: bool = True,
                 add_noise: bool = False,
                 noise_std: float = 0.01):
        """
        Initialize depth preprocessor
        
        Args:
            image_size: Target image size (square)
            depth_range: Valid depth range (min, max) in meters
            enable_denoising: Whether to apply denoising
            enable_enhancement: Whether to apply enhancement
            add_noise: Whether to add realistic noise
            noise_std: Standard deviation of added noise
        """
        self.image_size = image_size
        self.depth_min, self.depth_max = depth_range
        self.enable_denoising = enable_denoising and CV2_AVAILABLE
        self.enable_enhancement = enable_enhancement
        self.add_noise = add_noise
        self.noise_std = noise_std
        
        # Preprocessing statistics
        self.stats = {
            'processed_count': 0,
            'invalid_pixels': 0,
            'depth_range_violations': 0
        }
        
        print(f"🔧 Depth Preprocessor initialized:")
        print(f"   📏 Image size: {image_size}x{image_size}")
        print(f"   📊 Depth range: {depth_range[0]:.1f}m - {depth_range[1]:.1f}m")
        print(f"   🔇 Denoising: {'✅' if self.enable_denoising else '❌'}")
        print(f"   ✨ Enhancement: {'✅' if enable_enhancement else '❌'}")
        print(f"   🎲 Noise simulation: {'✅' if add_noise else '❌'}")
    
    def preprocess(self, depth_image: np.ndarray) -> torch.Tensor:
        """
        Apply full preprocessing pipeline to depth image
        
        Args:
            depth_image: Raw depth image (H, W) or (H, W, 1)
            
        Returns:
            Preprocessed depth tensor (1, 1, H, W) ready for CNN
        """
        # Ensure 2D array
        if depth_image.ndim == 3:
            depth_image = depth_image.squeeze()
        elif depth_image.ndim != 2:
            raise ValueError(f"Expected 2D or 3D depth image, got shape {depth_image.shape}")
        
        # Resize if needed
        if depth_image.shape != (self.image_size, self.image_size):
            depth_image = self._resize_image(depth_image)
        
        # Handle invalid values
        depth_image = self._handle_invalid_values(depth_image)
        
        # Apply denoising
        if self.enable_denoising:
            depth_image = self._denoise_image(depth_image)
        
        # Apply enhancement
        if self.enable_enhancement:
            depth_image = self._enhance_image(depth_image)
        
        # Add realistic noise
        if self.add_noise:
            depth_image = self._add_realistic_noise(depth_image)
        
        # Normalize depth values
        depth_image = self._normalize_depth(depth_image)
        
        # Convert to tensor and add batch/channel dimensions
        depth_tensor = torch.from_numpy(depth_image).float()
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        self.stats['processed_count'] += 1
        return depth_tensor
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size"""
        if CV2_AVAILABLE:
            return cv2.resize(image, (self.image_size, self.image_size), 
                            interpolation=cv2.INTER_LINEAR)
        else:
            # Simple resizing without external dependencies
            h, w = image.shape
            if h == self.image_size and w == self.image_size:
                return image
            
            # Simple bilinear interpolation
            y_indices = np.linspace(0, h-1, self.image_size)
            x_indices = np.linspace(0, w-1, self.image_size)
            
            # Create meshgrid for interpolation
            yi, xi = np.meshgrid(y_indices, x_indices, indexing='ij')
            
            # Simple nearest neighbor as fallback
            yi = np.round(yi).astype(int)
            xi = np.round(xi).astype(int)
            
            return image[yi, xi]
    
    def _handle_invalid_values(self, image: np.ndarray) -> np.ndarray:
        """Handle NaN, inf, and out-of-range values"""
        # Count invalid pixels
        invalid_mask = ~np.isfinite(image)
        out_of_range_mask = (image < self.depth_min) | (image > self.depth_max)
        
        self.stats['invalid_pixels'] += int(np.sum(invalid_mask))
        self.stats['depth_range_violations'] += int(np.sum(out_of_range_mask))
        
        # Replace invalid values with max depth
        image = np.where(invalid_mask, self.depth_max, image)
        
        # Clamp to valid range
        image = np.clip(image, self.depth_min, self.depth_max)
        
        return image
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising to reduce sensor noise"""
        if not CV2_AVAILABLE:
            return image
        
        # Convert to uint8 for OpenCV processing
        image_uint8 = ((image - self.depth_min) / (self.depth_max - self.depth_min) * 255).astype(np.uint8)
        
        # Apply Non-local Means denoising
        denoised_uint8 = cv2.fastNlMeansDenoising(image_uint8, None, 10, 7, 21)
        
        # Convert back to depth values
        denoised = denoised_uint8.astype(np.float32) / 255.0 * (self.depth_max - self.depth_min) + self.depth_min
        
        return denoised
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply enhancement to improve feature visibility"""
        # Apply edge-preserving smoothing
        if CV2_AVAILABLE:
            # Convert to uint8 for OpenCV
            image_uint8 = ((image - self.depth_min) / (self.depth_max - self.depth_min) * 255).astype(np.uint8)
            
            # Apply bilateral filter for edge-preserving smoothing
            enhanced_uint8 = cv2.bilateralFilter(image_uint8, 9, 75, 75)
            
            # Convert back to depth values
            enhanced = enhanced_uint8.astype(np.float32) / 255.0 * (self.depth_max - self.depth_min) + self.depth_min
            
            return enhanced
        else:
            # Simple smoothing without external dependencies
            # Apply a basic 3x3 averaging filter
            kernel = np.ones((3, 3)) / 9.0
            h, w = image.shape
            smoothed = np.zeros_like(image)
            
            for i in range(1, h-1):
                for j in range(1, w-1):
                    smoothed[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * kernel)
            
            # Handle borders by copying original values
            smoothed[0, :] = image[0, :]
            smoothed[-1, :] = image[-1, :]
            smoothed[:, 0] = image[:, 0]
            smoothed[:, -1] = image[:, -1]
            
            return smoothed
    
    def _add_realistic_noise(self, image: np.ndarray) -> np.ndarray:
        """Add realistic depth sensor noise"""
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_std, image.shape)
        
        # Add distance-dependent noise (further objects have more noise)
        distance_factor = (image - self.depth_min) / (self.depth_max - self.depth_min)
        distance_noise = np.random.normal(0, self.noise_std * distance_factor, image.shape)
        
        # Combine noises
        noisy_image = image + noise + distance_noise
        
        # Clamp to valid range
        return np.clip(noisy_image, self.depth_min, self.depth_max)
    
    def _normalize_depth(self, image: np.ndarray) -> np.ndarray:
        """Normalize depth values to [0, 1] range"""
        # Linear normalization
        normalized = (image - self.depth_min) / (self.depth_max - self.depth_min)
        
        # Apply logarithmic scaling for better feature distribution
        # This helps the CNN learn better from closer objects
        log_normalized = np.log1p(normalized * 10) / np.log1p(10)
        
        return log_normalized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        if self.stats['processed_count'] == 0:
            return self.stats
        
        return {
            **self.stats,
            'avg_invalid_pixels': self.stats['invalid_pixels'] / self.stats['processed_count'],
            'avg_range_violations': self.stats['depth_range_violations'] / self.stats['processed_count']
        }
    
    def reset_stats(self):
        """Reset preprocessing statistics"""
        self.stats = {
            'processed_count': 0,
            'invalid_pixels': 0,
            'depth_range_violations': 0
        }


class DepthAugmentor:
    """
    Data augmentation for depth images
    Useful for training CNN models
    """
    
    def __init__(self, 
                 enable_flip: bool = True,
                 enable_rotate: bool = True,
                 enable_scale: bool = True,
                 max_rotation: float = 5.0,
                 scale_range: Tuple[float, float] = (0.95, 1.05)):
        """
        Initialize depth augmentor
        
        Args:
            enable_flip: Enable horizontal flipping
            enable_rotate: Enable small rotations
            enable_scale: Enable scaling
            max_rotation: Maximum rotation angle in degrees
            scale_range: Scale factor range
        """
        self.enable_flip = enable_flip
        self.enable_rotate = enable_rotate and CV2_AVAILABLE
        self.enable_scale = enable_scale and CV2_AVAILABLE
        self.max_rotation = max_rotation
        self.scale_range = scale_range
        
        print(f"🎭 Depth Augmentor initialized:")
        print(f"   🔄 Flip: {'✅' if enable_flip else '❌'}")
        print(f"   🔄 Rotate: {'✅' if self.enable_rotate else '❌'}")
        print(f"   📏 Scale: {'✅' if self.enable_scale else '❌'}")
    
    def augment(self, depth_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to depth tensor
        
        Args:
            depth_tensor: Input depth tensor (1, 1, H, W)
            
        Returns:
            Augmented depth tensor
        """
        # Convert to numpy for processing
        depth_image = depth_tensor.squeeze().numpy()
        
        # Apply horizontal flip
        if self.enable_flip and np.random.random() > 0.5:
            depth_image = np.fliplr(depth_image)
        
        # Apply rotation
        if self.enable_rotate and np.random.random() > 0.5:
            angle = np.random.uniform(-self.max_rotation, self.max_rotation)
            depth_image = self._rotate_image(depth_image, angle)
        
        # Apply scaling
        if self.enable_scale and np.random.random() > 0.5:
            scale = np.random.uniform(*self.scale_range)
            depth_image = self._scale_image(depth_image, scale)
        
        # Convert back to tensor
        return torch.from_numpy(depth_image).float().unsqueeze(0).unsqueeze(0)
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle"""
        if not CV2_AVAILABLE:
            return image
        
        h, w = image.shape
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                               borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def _scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Scale image by given factor"""
        if not CV2_AVAILABLE:
            return image
        
        h, w = image.shape
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if scale > 1.0:
            # Crop from center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            scaled = scaled[start_h:start_h + h, start_w:start_w + w]
        else:
            # Pad to original size
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            scaled = np.pad(scaled, ((pad_h, h - new_h - pad_h), 
                                   (pad_w, w - new_w - pad_w)), 
                          mode='reflect')
        
        return scaled


def create_preprocessing_pipeline(config: Dict[str, Any]) -> DepthPreprocessor:
    """
    Create preprocessing pipeline from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured DepthPreprocessor
    """
    return DepthPreprocessor(
        image_size=config.get('depth_resolution', 64),
        depth_range=(0.1, config.get('depth_range', 10.0)),
        enable_denoising=config.get('enable_denoising', True),
        enable_enhancement=config.get('enable_enhancement', True),
        add_noise=config.get('add_noise', False),
        noise_std=config.get('noise_std', 0.01)
    )
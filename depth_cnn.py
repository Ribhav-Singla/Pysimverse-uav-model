"""
Depth Camera CNN Module
Implements convolutional neural network for processing MuJoCo depth camera images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple, Optional


class DepthCNN(nn.Module):
    """
    Convolutional Neural Network for processing depth camera images
    
    Architecture:
    - Input: (1, 64, 64) depth image
    - 3 convolutional layers with progressive feature extraction
    - Spatial attention mechanism for obstacle focus
    - Output: 128 features encoding spatial depth information
    """
    
    def __init__(self, input_size: int = 64, output_features: int = 128):
        super(DepthCNN, self).__init__()
        
        self.input_size = input_size
        self.output_features = output_features
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)  # 64x64 -> 32x32
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32x32 -> 16x16
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 16x16 -> 8x8
        self.bn3 = nn.BatchNorm2d(128)
        
        # Spatial attention mechanism
        self.attention = SpatialAttention(128)
        
        # Adaptive pooling to handle varying input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        fc_input_size = 128 * 4 * 4  # 2048
        self.fc1 = nn.Linear(fc_input_size, 512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, output_features)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the depth CNN
        
        Args:
            x: Input depth image tensor (batch_size, 1, height, width)
            
        Returns:
            Depth features tensor (batch_size, output_features)
        """
        # Convolutional feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Apply spatial attention
        x = self.attention(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected processing
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism to focus on important regions in depth images
    Helps the network pay more attention to obstacles and navigation-relevant areas
    """
    
    def __init__(self, channels: int):
        super(SpatialAttention, self).__init__()
        
        self.conv_attention = nn.Conv2d(channels, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to input features
        
        Args:
            x: Input feature tensor (batch_size, channels, height, width)
            
        Returns:
            Attention-weighted feature tensor
        """
        # Generate attention map
        attention_map = self.conv_attention(x)
        attention_weights = self.sigmoid(attention_map)
        
        # Apply attention weights
        attended_features = x * attention_weights
        
        return attended_features


class DepthFeatureExtractor:
    """
    High-level interface for extracting features from depth images using CNN
    Handles preprocessing, inference, and postprocessing
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = DepthCNN(input_size=64, output_features=128)
        self.model.to(self.device)
        
        # Load pretrained weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded pretrained depth CNN from {model_path}")
        else:
            print("🔄 Using randomly initialized depth CNN")
        
        self.model.eval()
        
        # Normalization parameters (will be learned from data)
        self.depth_mean = 0.5
        self.depth_std = 0.3
    
    def preprocess_depth_image(self, depth_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess depth image for CNN inference
        
        Args:
            depth_image: Raw depth image (height, width) in meters
            
        Returns:
            Preprocessed tensor ready for CNN
        """
        # Ensure correct shape
        if len(depth_image.shape) == 2:
            depth_image = depth_image[np.newaxis, ...]  # Add channel dimension
        
        # Normalize depth values to [0, 1] range
        depth_normalized = np.clip(depth_image / 5.0, 0, 1)  # Assume max 5m range
        
        # Apply standardization
        depth_standardized = (depth_normalized - self.depth_mean) / self.depth_std
        
        # Convert to tensor
        depth_tensor = torch.FloatTensor(depth_standardized).unsqueeze(0).to(self.device)
        
        return depth_tensor
    
    def extract_features(self, depth_input) -> np.ndarray:
        """
        Extract CNN features from depth image or tensor
        
        Args:
            depth_input: Raw depth image (height, width) in meters OR preprocessed tensor
            
        Returns:
            CNN-extracted features (128 dimensions)
        """
        # Handle both numpy arrays and tensors
        if isinstance(depth_input, torch.Tensor):
            # Input is already a preprocessed tensor
            depth_tensor = depth_input.to(self.device)
        else:
            # Input is a numpy array, need preprocessing
            depth_tensor = self.preprocess_depth_image(depth_input)
        
        # Extract features
        with torch.no_grad():
            features = self.model(depth_tensor)
            
        # Convert back to numpy
        features_np = features.cpu().numpy().squeeze()
        
        return features_np
    
    def get_spatial_features(self, depth_image: np.ndarray) -> dict:
        """
        Extract interpretable spatial features from depth image
        
        Args:
            depth_image: Raw depth image (height, width) in meters
            
        Returns:
            Dictionary of spatial features for analysis
        """
        features = {}
        
        # Basic statistics
        features['min_depth'] = np.min(depth_image)
        features['mean_depth'] = np.mean(depth_image)
        features['max_depth'] = np.max(depth_image)
        features['depth_std'] = np.std(depth_image)
        
        # Obstacle detection
        close_threshold = 1.0  # 1 meter
        features['close_obstacle_ratio'] = np.mean(depth_image < close_threshold)
        
        # Directional analysis (divide image into sectors)
        h, w = depth_image.shape
        center_h, center_w = h // 2, w // 2
        
        # Left/Right analysis
        left_half = depth_image[:, :center_w]
        right_half = depth_image[:, center_w:]
        features['left_clearance'] = np.mean(left_half)
        features['right_clearance'] = np.mean(right_half)
        
        # Top/Bottom analysis (for vertical obstacles)
        top_half = depth_image[:center_h, :]
        bottom_half = depth_image[center_h:, :]
        features['top_clearance'] = np.mean(top_half)
        features['bottom_clearance'] = np.mean(bottom_half)
        
        # Center region analysis (most important for navigation)
        center_region = depth_image[center_h-16:center_h+16, center_w-16:center_w+16]
        features['center_clearance'] = np.mean(center_region)
        
        return features


# Global instance for easy access
depth_feature_extractor = None

def get_depth_feature_extractor(model_path: Optional[str] = None) -> DepthFeatureExtractor:
    """Get or create global depth feature extractor instance"""
    global depth_feature_extractor
    
    if depth_feature_extractor is None:
        depth_feature_extractor = DepthFeatureExtractor(model_path)
    
    return depth_feature_extractor


if __name__ == "__main__":
    # Test the CNN architecture
    print("🧪 Testing Depth CNN Architecture...")
    
    # Create model
    model = DepthCNN(input_size=64, output_features=128)
    
    # Test with dummy data
    dummy_depth = torch.randn(1, 1, 64, 64)  # Batch of 1, single channel, 64x64
    
    with torch.no_grad():
        features = model(dummy_depth)
    
    print(f"✅ Input shape: {dummy_depth.shape}")
    print(f"✅ Output features shape: {features.shape}")
    print(f"✅ Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test feature extractor
    extractor = DepthFeatureExtractor()
    dummy_image = np.random.rand(64, 64) * 3.0  # Random depth image
    
    cnn_features = extractor.extract_features(dummy_image)
    spatial_features = extractor.get_spatial_features(dummy_image)
    
    print(f"✅ CNN features shape: {cnn_features.shape}")
    print(f"✅ Spatial features: {list(spatial_features.keys())}")
    print("🎉 Depth CNN architecture test completed successfully!")
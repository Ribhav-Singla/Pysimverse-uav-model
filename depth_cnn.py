import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional


class SimplDepthCNN(nn.Module):
    """
    Simple CNN for depth image processing without attention.
    2-3 CNN layers with ReLU activation, max pooling, and padding.
    
    Architecture:
    - Input: 1x64x64 depth image
    - Conv1: 1->16 channels, 3x3 kernel, ReLU, MaxPool
    - Conv2: 16->32 channels, 3x3 kernel, ReLU, MaxPool
    - Conv3: 32->64 channels, 3x3 kernel, ReLU, MaxPool
    - Flatten + MLP
    - Output: 128 features
    """
    
    def __init__(self, output_features: int = 128):
        super().__init__()
        self.output_features = output_features
        
        # Convolutional layers with padding
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 64x64 -> 64x64
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 32x32 -> 32x32
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 16x16 -> 16x16
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        
        # After pooling: 8x8 with 64 channels = 64*8*8 = 4096 features
        flattened_size = 64 * 8 * 8  # 4096
        
        # MLP layers
        self.fc1 = nn.Linear(flattened_size, 256)
        self.relu_fc1 = nn.ReLU(inplace=True)
        
        self.fc2 = nn.Linear(256, output_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 64, 64) or (64, 64) or numpy array
            
        Returns:
            Output tensor of shape (batch_size, output_features) or (output_features,)
        """
        # Handle numpy arrays
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Handle single 2D image (64x64)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            squeeze_output = True
        # Handle 3D (1x64x64) or add batch dim if needed
        elif x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dim
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Ensure tensor is on same device as model parameters
        device = next(self.parameters()).device
        x = x.to(device)
        
        # CNN layers
        x = self.pool1(self.relu1(self.conv1(x)))  # 64 -> 32
        x = self.pool2(self.relu2(self.conv2(x)))  # 32 -> 16
        x = self.pool3(self.relu3(self.conv3(x)))  # 16 -> 8
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, 4096)
        
        # MLP layers
        x = self.relu_fc1(self.fc1(x))  # (batch_size, 256)
        x = self.fc2(x)  # (batch_size, output_features)
        
        # Remove batch dimension if input was single image
        if squeeze_output:
            x = x.squeeze(0)
        
        return x


class DepthFeatureExtractor:
    """
    Wrapper class for extracting depth features from depth images using CNN.
    Handles both tensor and numpy array inputs.
    """
    
    def __init__(self, model_path: Optional[str] = None, output_features: int = 128):
        """
        Initialize depth feature extractor.
        
        Args:
            model_path: Path to pretrained CNN weights (if None, uses randomly initialized)
            output_features: Number of output features (default: 128)
        """
        self.output_features = output_features
        self.model = SimplDepthCNN(output_features=output_features)
        
        if model_path is not None:
            try:
                self.model.load_state_dict(torch.load(model_path))
                print(f"✅ Loaded pretrained CNN model from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load pretrained model from {model_path}: {e}")
                print("🔄 Using randomly initialized CNN model")
        else:
            print("🔄 Using randomly initialized CNN model")
        
        # Set to evaluation mode
        self.model.eval()
    
    def extract_features(self, depth_image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract CNN features from depth image.
        
        Args:
            depth_image: Input depth image as numpy array (64x64) or torch tensor
            
        Returns:
            Feature vector as numpy array of shape (output_features,)
        """
        # Convert to tensor if needed
        if isinstance(depth_image, np.ndarray):
            # Handle different input shapes
            if depth_image.ndim == 2:
                # (64, 64) -> (1, 1, 64, 64)
                depth_tensor = torch.from_numpy(depth_image).float().unsqueeze(0).unsqueeze(0)
            elif depth_image.ndim == 3:
                # (1, 64, 64) -> (1, 1, 64, 64)
                depth_tensor = torch.from_numpy(depth_image).float().unsqueeze(0)
            else:
                raise ValueError(f"Expected 2D or 3D depth image, got {depth_image.ndim}D")
        else:
            depth_tensor = depth_image
        
        # Extract features
        with torch.no_grad():
            features = self.model(depth_tensor)
        
        # Convert back to numpy
        if features.dim() > 1:
            features = features.squeeze(0)
        
        return features.cpu().numpy()
    
    def to_device(self, device: str):
        """Move model to specified device."""
        self.model = self.model.to(device)
    
    def get_model(self):
        """Get the underlying PyTorch model."""
        return self.model

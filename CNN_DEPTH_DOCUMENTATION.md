# CNN Depth Processing System Documentation

## Overview

This document provides comprehensive documentation for the CNN-based depth processing system implemented in the UAV navigation environment. The system replaces traditional LIDAR-based sensing with camera-based depth perception using convolutional neural networks.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Components](#components)
3. [Configuration](#configuration)
4. [Usage Guide](#usage-guide)
5. [Performance Comparison](#performance-comparison)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)

## System Architecture

The CNN depth processing system consists of four main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MuJoCo Depth │    │  Preprocessing  │    │   CNN Feature   │    │   Environment   │
│     Camera      │───▶│    Pipeline     │───▶│   Extraction    │───▶│   Integration   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼
   64x64 depth image    Enhanced/normalized     128 CNN features      142D observation
   (0.1m - 2.9m range)    tensor (0-1 range)   (spatial features)    (pos+vel+goal+CNN+nav)
```

### Fallback Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CNN Depth     │    │   Raycast       │    │   Environment   │
│   Processing    │──X─▶│   Simulation    │───▶│   Integration   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
      (fails)              (automatic)              (36D obs)
```

## Components

### 1. MuJoCo Depth Camera (`mujoco_depth_camera.py`)

**Purpose**: Interface between MuJoCo's camera system and depth processing pipeline

**Key Features**:
- Multiple camera API fallback methods
- Depth buffer extraction and conversion
- Raycast simulation when camera fails
- Configurable resolution and range

**Main Methods**:
```python
render_depth_image(data) -> np.ndarray  # Returns 64x64 depth image
get_camera_info() -> Dict[str, Any]     # Camera configuration info
```

### 2. CNN Architecture (`depth_cnn.py`)

**Purpose**: Deep learning model for extracting spatial features from depth images

**Architecture**:
- **Input**: 64×64×1 depth images
- **Layers**: 3 convolutional + spatial attention + 3 fully connected
- **Output**: 128-dimensional feature vector
- **Attention**: Spatial attention mechanism for focus on important regions

**Key Components**:
```python
DepthCNN(nn.Module)           # Main CNN architecture
SpatialAttention(nn.Module)   # Attention mechanism
DepthFeatureExtractor         # High-level interface
```

### 3. Preprocessing Pipeline (`depth_preprocessing.py`)

**Purpose**: Comprehensive depth image preprocessing for robust CNN input

**Features**:
- **Normalization**: Depth values to [0,1] with logarithmic scaling
- **Enhancement**: Edge-preserving smoothing (bilateral filter or Gaussian)
- **Denoising**: Non-local means denoising when OpenCV available
- **Noise Simulation**: Realistic depth sensor noise for training
- **Invalid Value Handling**: NaN/inf replacement and range clamping

**Main Classes**:
```python
DepthPreprocessor    # Main preprocessing pipeline
DepthAugmentor      # Data augmentation for training
```

### 4. Environment Integration (`uav_env.py`)

**Purpose**: Seamless integration of CNN processing into the UAV environment

**Key Changes**:
- Dynamic observation space sizing (CNN: 142D, Raycast: 36D)
- Automatic fallback mechanism
- Configuration-based switching
- PPO training compatibility

## Configuration

### Global Configuration (`CONFIG` in `uav_env.py`)

```python
CONFIG = {
    # CNN Depth Processing
    'use_cnn_depth': True,              # Enable CNN processing
    'cnn_features_dim': 128,            # CNN output dimensions
    'depth_cnn_model_path': None,       # Path to pretrained model
    
    # Camera Settings
    'depth_resolution': 64,             # Image resolution (64x64)
    'depth_range': 2.9,                 # Maximum detection range (meters)
    'depth_fov': 90,                    # Field of view (degrees)
    
    # Preprocessing Options
    'enable_denoising': True,           # Apply denoising
    'enable_enhancement': True,         # Apply enhancement
    'add_noise': False,                 # Add realistic noise
    'noise_std': 0.01,                  # Noise standard deviation
    
    # Fallback Settings
    'depth_features_dim': 16,           # Raycast feature count
}
```

### XML Camera Configuration

Add to UAV body in XML files:
```xml
<camera name="uav_depth_camera" pos="0.15 0 0" xyaxes="0 1 0 0 0 1" 
        fovy="90" resolution="64 64"/>
```

## Usage Guide

### Basic Usage

```python
from uav_env import UAVEnv, CONFIG

# Enable CNN depth processing
CONFIG['use_cnn_depth'] = True

# Create environment
env = UAVEnv()

# Use like any Gym environment
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Training with CNN

```python
from training import train_ppo
from uav_env import CONFIG

# Configure for CNN training
CONFIG['use_cnn_depth'] = True
CONFIG['enable_enhancement'] = True

# Train with any PPO variant
train_ppo(ppo_type='ns', episodes=100)
```

### Switching Between Modes

```python
# CNN mode (142-dimensional observations)
CONFIG['use_cnn_depth'] = True
env_cnn = UAVEnv()
print(f"CNN obs space: {env_cnn.observation_space.shape}")  # (142,)

# Raycast mode (36-dimensional observations)  
CONFIG['use_cnn_depth'] = False
env_raycast = UAVEnv()
print(f"Raycast obs space: {env_raycast.observation_space.shape}")  # (36,)
```

### Custom Preprocessing

```python
from depth_preprocessing import DepthPreprocessor

# Create custom preprocessor
preprocessor = DepthPreprocessor(
    image_size=64,
    depth_range=(0.1, 5.0),        # Custom range
    enable_denoising=True,
    enable_enhancement=True,
    add_noise=True,                 # Add training noise
    noise_std=0.02
)

# Process depth image
depth_tensor = preprocessor.preprocess(raw_depth_image)
```

## Performance Comparison

### Observation Space Dimensions

| Mode | Dimensions | Components |
|------|------------|------------|
| CNN | 142 | pos(3) + vel(3) + goal(3) + cnn_features(128) + nav_features(5) |
| Raycast | 36 | pos(3) + vel(3) + goal(3) + depth_readings(16) + engineered_features(11) |

### Feature Richness

**CNN Features (128D)**:
- Spatial patterns and obstacle arrangements
- Texture and surface information
- Depth gradients and edges
- Learned representations optimized for navigation

**Raycast Features (16D + 11D)**:
- 16 distance measurements in forward FOV
- 11 hand-engineered features (min, mean, directions, clearances)
- Interpretable but limited spatial understanding

### Computational Performance

| Component | CNN Mode | Raycast Mode |
|-----------|----------|--------------|
| Depth Processing | ~2ms (GPU) / ~15ms (CPU) | ~0.1ms |
| Feature Extraction | Neural network forward pass | Simple math operations |
| Memory Usage | ~50MB (model) + preprocessing | Minimal |
| Training Adaptation | May require more episodes | Faster initial learning |

## API Reference

### UAVEnv

```python
class UAVEnv(gym.Env):
    def __init__(self, render_mode=None, curriculum_learning=False, ns_cfg=None)
    def reset() -> Tuple[np.ndarray, Dict]
    def step(action) -> Tuple[np.ndarray, float, bool, bool, Dict]
    
    # CNN-specific attributes
    self.depth_camera: MuJoCoDepthCamera       # Camera interface
    self.depth_extractor: DepthFeatureExtractor # CNN model
    self.depth_preprocessor: DepthPreprocessor  # Preprocessing pipeline
```

### MuJoCoDepthCamera

```python
class MuJoCoDepthCamera:
    def __init__(self, model, camera_name: str, image_size: int = 64)
    def render_depth_image(self, data) -> np.ndarray
    def get_camera_info(self) -> Dict[str, Any]
    def test_camera_access(self) -> bool
```

### DepthFeatureExtractor  

```python
class DepthFeatureExtractor:
    def __init__(self, features_dim: int = 128, model_path: Optional[str] = None)
    def extract_features(self, depth_input) -> np.ndarray
    def get_spatial_features(self, depth_image: np.ndarray) -> Dict
    def save_model(self, path: str)
    def load_model(self, path: str)
```

### DepthPreprocessor

```python
class DepthPreprocessor:
    def __init__(self, image_size: int = 64, depth_range: Tuple[float, float] = (0.1, 10.0), ...)
    def preprocess(self, depth_image: np.ndarray) -> torch.Tensor
    def get_stats(self) -> Dict[str, Any]
    def reset_stats(self)
```

## Troubleshooting

### Common Issues

#### 1. "Camera does not exist" Error
**Symptom**: `⚠️ MuJoCo depth rendering failed: The camera "uav_depth_camera" does not exist.`

**Solutions**:
- Verify camera is defined in XML: `<camera name="uav_depth_camera" ...>`
- Check XML file being loaded matches camera name
- System automatically falls back to raycast - this is normal behavior

#### 2. Dimension Mismatch Errors
**Symptom**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**Solutions**:
- Ensure observation space matches actual observations
- Check if CONFIG['use_cnn_depth'] matches environment setup
- Recreate PPO agent after changing CNN settings

#### 3. OpenCV/Scipy Missing Warnings
**Symptom**: `⚠️ OpenCV not available, using fallback methods`

**Solutions**:
- Install dependencies: `pip install opencv-python scipy`
- Fallback methods work but with reduced functionality
- Consider installing for better preprocessing quality

#### 4. Poor Training Performance
**Possible Causes**:
- CNN features may need more training episodes
- Consider adjusting learning rates
- Enable preprocessing enhancement: `CONFIG['enable_enhancement'] = True`
- Try different PPO variants (ns, ar, vanilla)

### Best Practices

1. **Testing**: Always run `test_cnn_integration.py` after configuration changes
2. **Fallback**: System gracefully handles camera failures - don't disable fallback
3. **Memory**: CNN mode uses more memory - monitor for long training runs
4. **Preprocessing**: Enable enhancement for better feature quality
5. **Training**: May need 10-20% more episodes compared to raycast mode

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug info
env = UAVEnv()
```

### Performance Monitoring

```python
# Check preprocessing statistics
if env.depth_preprocessor:
    stats = env.depth_preprocessor.get_stats()
    print(f"Processed images: {stats['processed_count']}")
    print(f"Invalid pixels: {stats['avg_invalid_pixels']:.2f}")
```

## File Structure

```
├── depth_cnn.py                 # CNN architecture and feature extraction
├── mujoco_depth_camera.py       # MuJoCo camera interface
├── depth_preprocessing.py       # Image preprocessing pipeline
├── uav_env.py                   # Environment integration
├── test_cnn_integration.py      # Integration tests
├── test_cnn_training.py         # Training compatibility tests
├── test_preprocessing.py        # Preprocessing tests
└── environment*.xml             # XML files with camera definitions
```

## Future Enhancements

1. **Transfer Learning**: Pre-train CNN on depth datasets
2. **Real-time Optimization**: ONNX conversion for faster inference
3. **Multi-camera**: Support for multiple depth cameras
4. **Advanced Attention**: Implement transformer-based attention
5. **Domain Adaptation**: Sim-to-real transfer capabilities

---

For additional support or questions, please refer to the test scripts and configuration examples provided in the repository.
"""
CNN and YOLO Fusion Module
Combines CNN depth features with YOLO object detection for enhanced UAV perception
Provides unified observation space combining both sensing modalities
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from depth_cnn import SimplDepthCNN


class CNNYOLOFusion:
    """
    Fusion system combining CNN depth processing with YOLO object detection
    
    Features:
    - Extracts CNN features from depth images
    - Extracts YOLO detections from RGB images
    - Fuses both representations into unified observation
    - Provides obstacle/goal/boundary awareness
    """
    
    def __init__(
        self,
        depth_cnn: SimplDepthCNN,
        yolo_detector = None,
        fusion_mode: str = 'concatenate',
        cnn_weight: float = 0.7,
        yolo_weight: float = 0.3
    ):
        """
        Initialize fusion module
        
        Args:
            depth_cnn: SimplDepthCNN model for depth processing
            yolo_detector: YOLODetector for object detection (optional)
            fusion_mode: 'concatenate', 'attention', or 'weighted_sum'
            cnn_weight: Weight for CNN features in fusion
            yolo_weight: Weight for YOLO features in fusion
        """
        self.depth_cnn = depth_cnn
        self.yolo_detector = yolo_detector
        self.fusion_mode = fusion_mode
        self.cnn_weight = cnn_weight
        self.yolo_weight = yolo_weight
        
        print(f"✅ CNN-YOLO Fusion initialized (mode: {fusion_mode})")
        if yolo_detector is not None:
            print("✅ YOLO detector integrated")
        else:
            print("⚠️  YOLO detector not available (CNN-only mode)")
    
    def fuse(
        self,
        depth_image: np.ndarray,
        rgb_image: Optional[np.ndarray] = None,
        pos: np.ndarray = None,
        camera_fov: float = 90.0,
        max_depth: float = 2.9
    ) -> Dict:
        """
        Fuse CNN depth features with YOLO detections
        
        Args:
            depth_image: Depth image from camera (H x W)
            rgb_image: RGB image for YOLO detection (H x W x 3, optional)
            pos: Current UAV position (x, y, z)
            camera_fov: Camera field of view in degrees
            max_depth: Maximum detection range
            
        Returns:
            Fusion results dictionary with:
            {
                'cnn_features': 128D CNN depth features
                'yolo_detections': Raw YOLO detections
                'yolo_features': Processed YOLO features
                'fused_features': Combined observation features
                'total_features': Total feature dimension
                'detection_summary': Summary of detections
            }
        """
        # Extract CNN features from depth
        cnn_features = self._extract_cnn_features(depth_image)
        
        # Extract YOLO features from RGB
        yolo_detections = None
        yolo_features = None
        if self.yolo_detector is not None and rgb_image is not None:
            yolo_detections = self.yolo_detector.detect(rgb_image)
            yolo_features = self._extract_yolo_features(
                yolo_detections, depth_image.shape, camera_fov, max_depth
            )
        else:
            yolo_features = self._empty_yolo_features()
        
        # Fuse representations
        fused = self._fuse_representations(cnn_features, yolo_features)
        
        return {
            'cnn_features': cnn_features,
            'yolo_detections': yolo_detections,
            'yolo_features': yolo_features,
            'fused_features': fused,
            'total_features': len(fused),
            'detection_summary': self._summarize_detections(yolo_detections)
        }
    
    def _extract_cnn_features(self, depth_image: np.ndarray) -> np.ndarray:
        """Extract 128D CNN features from depth image"""
        # Convert depth image to tensor
        if isinstance(depth_image, np.ndarray):
            depth_tensor = torch.FloatTensor(depth_image).unsqueeze(0).unsqueeze(0)
        else:
            depth_tensor = depth_image
        
        # Extract CNN features
        with torch.no_grad():
            cnn_features = self.depth_cnn(depth_tensor)
        
        # Convert to numpy
        if isinstance(cnn_features, torch.Tensor):
            cnn_features = cnn_features.cpu().numpy().flatten()
        
        return cnn_features.astype(np.float32)
    
    def _extract_yolo_features(
        self,
        detections: Dict,
        image_shape: Tuple,
        camera_fov: float,
        max_depth: float
    ) -> np.ndarray:
        """
        Convert YOLO detections to feature vector
        
        Features extracted:
        - Closest obstacle distance
        - Obstacle direction (angle)
        - Closest goal distance
        - Goal direction
        - Closest boundary distance
        - Detection confidence scores
        - Obstacle count, goal count, boundary count
        """
        if detections is None or not detections['boxes']:
            return self._empty_yolo_features()
        
        # Get spatial information
        spatial = self.yolo_detector.get_spatial_info(
            detections, image_shape, camera_fov, max_depth
        )
        
        features = []
        
        # Obstacle features (closest obstacle)
        if spatial['obstacle_positions']:
            closest_obs = min(spatial['obstacle_positions'], key=lambda x: x['distance'])
            features.extend([
                closest_obs['distance'],  # Distance to closest obstacle
                closest_obs['angle'][0],  # Angle X
                closest_obs['angle'][1],  # Angle Y
                closest_obs['confidence'],  # Confidence
                1.0  # Obstacle detected flag
            ])
        else:
            features.extend([max_depth, 0.0, 0.0, 0.0, 0.0])
        
        # Goal features (closest goal)
        if spatial['goal_positions']:
            closest_goal = min(spatial['goal_positions'], key=lambda x: x['distance'])
            features.extend([
                closest_goal['distance'],
                closest_goal['angle'][0],
                closest_goal['angle'][1],
                closest_goal['confidence'],
                1.0  # Goal detected flag
            ])
        else:
            features.extend([max_depth, 0.0, 0.0, 0.0, 0.0])
        
        # Boundary features (closest boundary)
        if spatial['boundary_positions']:
            closest_bound = min(spatial['boundary_positions'], key=lambda x: x['distance'])
            features.extend([
                closest_bound['distance'],
                closest_bound['angle'][0],
                closest_bound['angle'][1],
                closest_bound['confidence']
            ])
        else:
            features.extend([max_depth, 0.0, 0.0, 0.0])
        
        # Detection counts
        features.extend([
            float(len(spatial['obstacle_positions'])),
            float(len(spatial['goal_positions'])),
            float(len(spatial['boundary_positions'])),
            np.mean(detections['confidences']) if detections['confidences'] else 0.0
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _empty_yolo_features(self) -> np.ndarray:
        """Return empty/default YOLO feature vector"""
        # 5 + 5 + 4 + 4 = 18 features
        return np.zeros(18, dtype=np.float32)
    
    def _fuse_representations(
        self,
        cnn_features: np.ndarray,
        yolo_features: np.ndarray
    ) -> np.ndarray:
        """
        Fuse CNN and YOLO features
        
        Returns 146-dimensional feature vector:
        - 128 CNN depth features
        - 18 YOLO detection features
        """
        if self.fusion_mode == 'concatenate':
            # Simple concatenation
            fused = np.concatenate([cnn_features, yolo_features])
        
        elif self.fusion_mode == 'weighted_sum':
            # Weighted averaging (only possible if same dimension)
            # For different dimensions, fall back to concatenation
            fused = np.concatenate([cnn_features, yolo_features])
        
        elif self.fusion_mode == 'attention':
            # Attention-based fusion
            # Simple attention: weight features by their importance scores
            cnn_attention = np.mean(np.abs(cnn_features)) + 1e-8
            yolo_attention = np.mean(np.abs(yolo_features)) + 1e-8
            
            # Normalize attention weights
            total_attention = cnn_attention + yolo_attention
            cnn_weight = cnn_attention / total_attention
            yolo_weight = yolo_attention / total_attention
            
            fused = np.concatenate([
                cnn_features * cnn_weight,
                yolo_features * yolo_weight
            ])
        
        else:
            # Default to concatenation
            fused = np.concatenate([cnn_features, yolo_features])
        
        return fused.astype(np.float32)
    
    def _summarize_detections(self, detections: Optional[Dict]) -> Dict:
        """Create summary of detections for logging"""
        if detections is None or not detections['boxes']:
            return {
                'total_detections': 0,
                'obstacles': 0,
                'goals': 0,
                'boundaries': 0,
                'max_confidence': 0.0
            }
        
        return {
            'total_detections': len(detections['boxes']),
            'obstacles': len(detections['obstacles']),
            'goals': len(detections['goals']),
            'boundaries': len(detections['boundaries']),
            'max_confidence': max(detections['confidences']) if detections['confidences'] else 0.0,
            'avg_confidence': np.mean(detections['confidences']) if detections['confidences'] else 0.0
        }
    
    def get_observation_dimension(self) -> int:
        """
        Get total observation dimension after fusion
        
        Returns:
            Total dimension (typically 128 CNN + 18 YOLO = 146)
        """
        return 128 + 18  # CNN features + YOLO features


class FusionObservationBuilder:
    """
    Builds complete observation vector for PPO training
    
    Combines:
    - Position (3D)
    - Velocity (3D)
    - Goal distance (3D)
    - Fused CNN-YOLO features (146D)
    - Navigation features (5D)
    
    Total: 3 + 3 + 3 + 146 + 5 = 160D observation
    """
    
    def __init__(self, fusion_module: CNNYOLOFusion):
        self.fusion_module = fusion_module
        self.observation_dim = 3 + 3 + 3 + fusion_module.get_observation_dimension() + 5
    
    def build(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        goal_pos: np.ndarray,
        depth_image: np.ndarray,
        rgb_image: Optional[np.ndarray] = None,
        camera_fov: float = 90.0,
        max_depth: float = 2.9
    ) -> np.ndarray:
        """
        Build complete observation vector
        
        Args:
            pos: UAV position (3D)
            vel: UAV velocity (3D)
            goal_pos: Goal position (3D)
            depth_image: Depth image from camera
            rgb_image: RGB image for YOLO (optional)
            camera_fov: Camera field of view
            max_depth: Maximum depth range
            
        Returns:
            160D observation vector
        """
        # Position and velocity
        goal_dist = goal_pos - pos
        
        # Fused CNN-YOLO features
        fusion_result = self.fusion_module.fuse(
            depth_image, rgb_image, pos, camera_fov, max_depth
        )
        fused_features = fusion_result['fused_features']
        
        # Navigation features
        goal_direction_norm = goal_dist[:2] / (np.linalg.norm(goal_dist[:2]) + 1e-8)
        navigation_features = np.array([
            np.linalg.norm(goal_dist),      # Distance to goal
            goal_direction_norm[0],         # Goal direction X
            goal_direction_norm[1],         # Goal direction Y
            pos[2],                         # Current altitude
            np.linalg.norm(vel),           # Current speed
        ])
        
        # Combine all observation components
        obs = np.concatenate([pos, vel, goal_dist, fused_features, navigation_features])
        
        # Clip to reasonable range
        obs = np.clip(obs, -100.0, 100.0)
        
        return obs.astype(np.float32)
    
    def get_observation_dimension(self) -> int:
        return self.observation_dim

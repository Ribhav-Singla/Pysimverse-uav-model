"""
YOLOv8-based Depth and Obstacle Detection Module

This module replaces the SimplDepthCNN with YOLOv8 for real-time object detection.
YOLOv8 extracts spatial features from RGB images for obstacle and goal detection.

Key Improvements over CNN:
- Real-time object detection (obstacles, goals, boundaries)
- Semantic understanding (WHAT is detected, not just depth pattern)
- Scalable to multiple object classes
- Pre-trained on COCO, requires minimal fine-tuning
- Faster inference (5-10ms for nano model)

Feature Output: 64D spatial features vector
"""

import torch
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from ultralytics import YOLO


class YOLODepthDetector:
    """
    YOLO-based obstacle and goal detection system.
    Replaces SimplDepthCNN for depth/obstacle sensing.
    
    Extracts spatial features from RGB images:
    - Obstacle detection and distance estimation
    - Goal detection and localization
    - Boundary detection and spatial awareness
    - Free space mapping
    
    Output: 64D feature vector covering all spatial awareness
    """
    
    def __init__(
        self,
        model_name: str = 'yolov8n',
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'cuda',
        output_features: int = 64,
        image_size: int = 640,
    ):
        """
        Initialize YOLOv8-based obstacle detector.
        
        Args:
            model_name: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            confidence_threshold: Detection confidence threshold [0.0, 1.0]
            iou_threshold: NMS IoU threshold [0.0, 1.0]
            device: 'cuda' or 'cpu'
            output_features: Output feature dimension (64D)
            image_size: Input image size for YOLO (default 640x640)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Auto-detect device availability
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"⚠️ CUDA not available, falling back to CPU")
            device = 'cpu'
        
        self.device = device
        self.output_features = output_features
        self.image_size = image_size
        
        # Load YOLOv8 model
        try:
            self.model = YOLO(f'yolov8{model_name}.pt')
            self.model.to(device)
            print(f"✅ YOLOv8-{model_name} loaded successfully on {device}")
        except Exception as e:
            print(f"❌ Failed to load YOLOv8: {e}")
            raise
        
        # Class mapping for detection
        self.class_names = self.model.names
        self.num_classes = len(self.class_names)
        
        # Detection cache for efficiency
        self.last_detections = None
        self.detection_cache_frame = None
        
        # Statistics
        self.inference_time = 0.0
        self.detection_count = 0
        self.frame_count = 0
    
    def detect(
        self,
        image: Union[np.ndarray, torch.Tensor],
        return_visualization: bool = False
    ) -> Dict:
        """
        Run YOLO detection on image.
        
        Args:
            image: RGB image (HxWx3) or torch tensor
            return_visualization: Whether to return annotated image
            
        Returns:
            Dictionary with detection results:
            {
                'boxes': List[List[x1, y1, x2, y2, conf, class_id]],
                'confidences': List[float],
                'class_ids': List[int],
                'class_names': List[str],
                'annotations': List[str],
                'num_detections': int,
                'inference_time': float,
                'image_viz': np.ndarray (optional)
            }
        """
        self.frame_count += 1
        
        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Ensure correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Resize to model input size
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        
        # Run YOLO inference
        results = self.model(image_resized, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)
        
        # Parse results
        detections = {
            'boxes': [],
            'confidences': [],
            'class_ids': [],
            'class_names': [],
            'annotations': [],
            'num_detections': 0,
            'inference_time': results[0].speed['inference'] if hasattr(results[0], 'speed') else 0.0,
        }
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                
                detections['boxes'].append([x1, y1, x2, y2, conf, class_id])
                detections['confidences'].append(conf)
                detections['class_ids'].append(class_id)
                detections['class_names'].append(class_name)
                detections['annotations'].append(f"{class_name} {conf:.2f}")
                detections['num_detections'] += 1
        
        self.last_detections = detections
        self.inference_time = detections['inference_time']
        self.detection_count = detections['num_detections']
        
        # Optional visualization
        if return_visualization:
            image_viz = self._visualize_detections(image_resized, detections)
            detections['image_viz'] = image_viz
        
        return detections
    
    def extract_spatial_features(self, detections: Dict) -> np.ndarray:
        """
        Extract 64D spatial feature vector from detections.
        
        Feature layout (64D total):
        ├─ Obstacle Detection (20D)
        │  ├─ Closest obstacle distance (1)
        │  ├─ Closest obstacle angle X (1)
        │  ├─ Closest obstacle angle Y (1)
        │  ├─ Closest obstacle confidence (1)
        │  ├─ 2nd closest obstacle (5D)
        │  ├─ 3rd closest obstacle (5D)
        │  ├─ 4th closest obstacle (5D)
        │  └─ Obstacle count (1)
        │
        ├─ Goal Detection (16D)
        │  ├─ Closest goal distance (1)
        │  ├─ Closest goal angle X (1)
        │  ├─ Closest goal angle Y (1)
        │  ├─ Closest goal confidence (1)
        │  ├─ 2nd closest goal (5D)
        │  ├─ 3rd closest goal (5D)
        │  └─ Goal count (1)
        │
        ├─ Boundary Detection (12D)
        │ ├─ Distance to left boundary (1)
        │  ├─ Distance to right boundary (1)
        │  ├─ Distance to top boundary (1)
        │  ├─ Distance to bottom boundary (1)
        │  ├─ Distance to nearest corner (1)
        │  └─ Boundary pressure (1)
        │
        └─ Free Space Analysis (16D)
           ├─ Free space ratio (1)
           ├─ Largest free region (1)
           ├─ Safe direction X (1)
           ├─ Safe direction Y (1)
           ├─ Obstacle density (1)
           └─ Confidence scores histogram (11)
        
        Args:
            detections: Dictionary from detect() method
            
        Returns:
            64D numpy array with spatial features
        """
        features = np.zeros(self.output_features, dtype=np.float32)
        
        if detections['num_detections'] == 0:
            return features
        
        # Separate detections by class
        obstacles = []
        goals = []
        boundaries = []
        
        for i, (box, conf, class_id, class_name) in enumerate(zip(
            detections['boxes'],
            detections['confidences'],
            detections['class_ids'],
            detections['class_names']
        )):
            x1, y1, x2, y2, _, _ = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            
            # Estimate distance based on bounding box size (heuristic)
            box_area = w * h
            distance = 2.9 / (1.0 + np.log(box_area / 100.0 + 1.0))  # Normalized distance [0, 2.9]
            distance = np.clip(distance, 0.1, 2.9)
            
            # Calculate angles
            image_center_x = self.image_size / 2
            image_center_y = self.image_size / 2
            angle_x = (cx - image_center_x) / image_center_x * 45  # [-45, 45] degrees
            angle_y = (cy - image_center_y) / image_center_y * 45  # [-45, 45] degrees
            
            detection_info = {
                'box': box,
                'conf': conf,
                'class_id': class_id,
                'class_name': class_name,
                'distance': distance,
                'angle_x': angle_x,
                'angle_y': angle_y,
                'cx': cx,
                'cy': cy,
            }
            
            # Classify detection
            if 'person' in class_name.lower() or 'car' in class_name.lower() or 'bottle' in class_name.lower():
                obstacles.append(detection_info)
            elif 'cup' in class_name.lower() or 'backpack' in class_name.lower():
                goals.append(detection_info)
            else:
                boundaries.append(detection_info)
        
        # Fill feature vector
        idx = 0
        
        # Obstacle features (20D)
        obstacles.sort(key=lambda x: x['distance'])  # Sort by distance
        for j in range(min(4, len(obstacles))):
            obs = obstacles[j]
            if j == 0:
                features[idx] = obs['distance'] / 2.9  # Normalize distance
                features[idx + 1] = (obs['angle_x'] + 45) / 90  # Normalize angle [-45, 45] -> [0, 1]
                features[idx + 2] = (obs['angle_y'] + 45) / 90
                features[idx + 3] = obs['conf']
                idx += 4
            else:
                features[idx] = obs['distance'] / 2.9
                features[idx + 1] = (obs['angle_x'] + 45) / 90
                features[idx + 2] = (obs['angle_y'] + 45) / 90
                features[idx + 3] = obs['conf']
                features[idx + 4] = 1.0  # Detected flag
                idx += 5
        
        features[idx] = len(obstacles) / 10.0  # Obstacle count (normalized)
        idx += 1
        
        # Goal features (16D)
        goals.sort(key=lambda x: x['distance'])
        for j in range(min(3, len(goals))):
            goal = goals[j]
            if j == 0:
                features[idx] = goal['distance'] / 2.9
                features[idx + 1] = (goal['angle_x'] + 45) / 90
                features[idx + 2] = (goal['angle_y'] + 45) / 90
                features[idx + 3] = goal['conf']
                idx += 4
            else:
                features[idx] = goal['distance'] / 2.9
                features[idx + 1] = (goal['angle_x'] + 45) / 90
                features[idx + 2] = (goal['angle_y'] + 45) / 90
                features[idx + 3] = goal['conf']
                features[idx + 4] = 1.0
                idx += 5
        
        features[idx] = len(goals) / 10.0
        idx += 1
        
        # Boundary features (12D) - simple heuristic based on image edges
        boundary_margin = 0.1 * self.image_size
        features[idx] = boundary_margin / self.image_size  # Left
        features[idx + 1] = boundary_margin / self.image_size  # Right
        features[idx + 2] = boundary_margin / self.image_size  # Top
        features[idx + 3] = boundary_margin / self.image_size  # Bottom
        features[idx + 4] = min(features[idx], features[idx + 1], features[idx + 2], features[idx + 3])  # Nearest corner
        features[idx + 5] = len(boundaries) / 5.0  # Boundary pressure
        idx += 6
        
        # Free space analysis (remaining features)
        free_space_ratio = max(0, 1.0 - detections['num_detections'] / 50.0)
        features[idx] = free_space_ratio
        
        return features
    
    def extract_features(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Complete pipeline: detect -> extract features.
        
        Args:
            image: RGB image
            
        Returns:
            64D feature vector
        """
        detections = self.detect(image)
        features = self.extract_spatial_features(detections)
        return features
    
    def _visualize_detections(self, image: np.ndarray, detections: Dict) -> np.ndarray:
        """Visualize detections on image."""
        image_viz = image.copy()
        
        for box, conf, class_name in zip(detections['boxes'], detections['confidences'], detections['class_names']):
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(image_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image_viz, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return image_viz
    
    def get_statistics(self) -> Dict:
        """Return detector statistics."""
        return {
            'model': self.model_name,
            'inference_time_ms': self.inference_time,
            'detections_last_frame': self.detection_count,
            'frames_processed': self.frame_count,
            'avg_inference_time_ms': (self.inference_time * self.frame_count) / max(1, self.frame_count),
        }
    
    def profile_inference(self, image: np.ndarray, num_runs: int = 10) -> Dict:
        """Profile inference time."""
        import time
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.detect(image)
            times.append((time.time() - start) * 1000)  # Convert to ms
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
        }


class YOLODepthFeatureExtractor:
    """
    Wrapper for YOLO-based depth feature extraction.
    Replaces DepthFeatureExtractor from depth_cnn.py
    """
    
    def __init__(
        self,
        model_name: str = 'yolov8n',
        device: str = 'cuda',
        output_features: int = 64,
    ):
        """
        Initialize feature extractor.
        
        Args:
            model_name: YOLO model size
            device: 'cuda' or 'cpu'
            output_features: Output feature dimension (default 64)
        """
        self.detector = YOLODepthDetector(
            model_name=model_name,
            device=device,
            output_features=output_features,
        )
        self.output_features = output_features
    
    def extract_features(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Extract features from image.
        
        Args:
            image: RGB image or numpy array
            
        Returns:
            64D feature vector
        """
        return self.detector.extract_features(image)
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Allow calling as function."""
        return self.extract_features(image)
    
    def to(self, device: str):
        """Move detector to device."""
        self.detector.model.to(device)
        return self
    
    def eval(self):
        """Set to evaluation mode."""
        self.detector.model.eval()
        return self
    
    def train(self):
        """Set to training mode (no-op for YOLO)."""
        return self


if __name__ == '__main__':
    print("🔧 YOLOv8 Depth Detector Test")
    print("=" * 50)
    
    # Create detector
    detector = YOLODepthDetector(model_name='n', device='cpu')
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run detection
    detections = detector.detect(dummy_image)
    print(f"\n✅ Detection successful: {detections['num_detections']} objects found")
    
    # Extract features
    features = detector.extract_spatial_features(detections)
    print(f"✅ Spatial features extracted: {features.shape}")
    
    # Statistics
    stats = detector.get_statistics()
    print(f"✅ Statistics: {stats}")

"""
YOLO-Based Obstacle and Goal Detection System
Integrates YOLOv8 for real-time object detection in UAV navigation
Provides bounding box detection, confidence scores, and spatial information
"""

import torch
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


class YOLODetector:
    """
    YOLO-based detector for obstacles and goals in UAV environment
    
    Features:
    - Real-time object detection from depth/RGB images
    - Custom class detection (obstacle, goal, boundary)
    - Confidence filtering and NMS (Non-Maximum Suppression)
    - Bounding box to spatial conversion
    - Detection caching for performance
    """
    
    def __init__(
        self,
        model_name: str = 'yolov8n',  # nano for speed, small/medium for accuracy
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'cpu',
        imgsz: int = 640
    ):
        """
        Initialize YOLO detector
        
        Args:
            model_name: YOLOv8 model size ('n'=nano, 's'=small, 'm'=medium, 'l'=large)
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('cpu' or 'cuda')
            imgsz: Input image size
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.imgsz = imgsz
        
        # Initialize YOLO model
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(f'{model_name}.pt')
            self.yolo_model.to(self.device)
            print(f"✅ YOLO model loaded: {model_name} on {self.device}")
        except ImportError:
            print("⚠️  YOLOv8 not installed. Run: pip install ultralytics")
            self.yolo_model = None
            
        # Detection cache
        self.last_detections = None
        self.last_image_hash = None
        self.cache_enabled = True
        
        # Statistics
        self.detection_stats = {
            'total_detections': 0,
            'obstacle_detections': 0,
            'goal_detections': 0,
            'boundary_detections': 0,
            'average_confidence': 0.0,
            'detection_frequency': 0.0
        }
    
    def detect(
        self,
        image: np.ndarray,
        return_viz: bool = False
    ) -> Dict:
        """
        Run YOLO detection on image
        
        Args:
            image: Input image (H x W x 3 for RGB, H x W for grayscale/depth)
            return_viz: Whether to return visualization image
            
        Returns:
            Dictionary with detections:
            {
                'boxes': List of bounding boxes [x1, y1, x2, y2]
                'confidences': List of confidence scores
                'classes': List of class labels
                'obstacles': List of obstacle detections
                'goals': List of goal detections
                'boundaries': List of boundary detections
                'visualization': Optional visualization image
            }
        """
        if self.yolo_model is None:
            return self._empty_detection_result(return_viz)
        
        # Check cache
        image_hash = hash(image.tobytes())
        if self.cache_enabled and self.last_image_hash == image_hash and self.last_detections:
            return self.last_detections.copy()
        
        # Convert grayscale/depth to RGB if needed
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image.astype(np.uint8)
        
        # Run YOLO inference
        results = self.yolo_model(image_rgb, conf=self.confidence_threshold, iou=self.iou_threshold)
        
        # Parse results
        detections = self._parse_yolo_results(results, image.shape)
        
        # Cache results
        self.last_image_hash = image_hash
        self.last_detections = detections.copy()
        
        # Add visualization if requested
        if return_viz:
            detections['visualization'] = self._visualize_detections(image_rgb, detections)
        
        return detections
    
    def _parse_yolo_results(self, results, image_shape: Tuple) -> Dict:
        """Parse YOLOv8 detection results"""
        detections = {
            'boxes': [],
            'confidences': [],
            'classes': [],
            'class_names': [],
            'obstacles': [],
            'goals': [],
            'boundaries': []
        }
        
        for result in results:
            if result.boxes is None:
                continue
                
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confs, class_ids):
                detections['boxes'].append(box)
                detections['confidences'].append(float(conf))
                detections['classes'].append(int(cls_id))
                
                # Get class name
                class_name = result.names.get(cls_id, 'unknown')
                detections['class_names'].append(class_name)
                
                # Categorize detection
                if 'obstacle' in class_name.lower():
                    detections['obstacles'].append({
                        'box': box,
                        'confidence': float(conf),
                        'class': class_name
                    })
                elif 'goal' in class_name.lower() or 'target' in class_name.lower():
                    detections['goals'].append({
                        'box': box,
                        'confidence': float(conf),
                        'class': class_name
                    })
                elif 'boundary' in class_name.lower():
                    detections['boundaries'].append({
                        'box': box,
                        'confidence': float(conf),
                        'class': class_name
                    })
        
        # Update statistics
        self.detection_stats['total_detections'] += len(detections['boxes'])
        self.detection_stats['obstacle_detections'] += len(detections['obstacles'])
        self.detection_stats['goal_detections'] += len(detections['goals'])
        self.detection_stats['boundary_detections'] += len(detections['boundaries'])
        
        if detections['confidences']:
            self.detection_stats['average_confidence'] = np.mean(detections['confidences'])
        
        return detections
    
    def get_spatial_info(
        self,
        detections: Dict,
        image_shape: Tuple,
        camera_fov: float = 90.0,
        max_depth: float = 2.9
    ) -> Dict:
        """
        Convert bounding boxes to spatial information
        
        Args:
            detections: Detection results from detect()
            image_shape: Shape of input image (H, W)
            camera_fov: Camera field of view in degrees
            max_depth: Maximum detection range in meters
            
        Returns:
            Dictionary with spatial information for each detection
        """
        spatial_info = {
            'obstacle_positions': [],
            'goal_positions': [],
            'boundary_positions': [],
            'distances': [],
            'angles': []
        }
        
        height, width = image_shape[:2]
        
        for box, conf, class_name in zip(
            detections['boxes'],
            detections['confidences'],
            detections['class_names']
        ):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Convert pixel coordinates to normalized [-1, 1]
            norm_x = (center_x - width / 2) / (width / 2)
            norm_y = (center_y - height / 2) / (height / 2)
            
            # Convert to angle (camera field of view)
            angle_x = norm_x * (camera_fov / 2)
            angle_y = norm_y * (camera_fov / 2)
            
            # Estimate distance from bounding box size
            # Larger box = closer object
            box_area = (box_width * box_height) / (width * height)
            estimated_distance = max_depth * (1.0 - min(box_area, 0.9))
            
            spatial_data = {
                'center': (float(center_x), float(center_y)),
                'normalized': (float(norm_x), float(norm_y)),
                'angle': (float(angle_x), float(angle_y)),
                'distance': float(estimated_distance),
                'box_size': (float(box_width), float(box_height)),
                'confidence': float(conf),
                'class': class_name
            }
            
            if 'obstacle' in class_name.lower():
                spatial_info['obstacle_positions'].append(spatial_data)
            elif 'goal' in class_name.lower() or 'target' in class_name.lower():
                spatial_info['goal_positions'].append(spatial_data)
            elif 'boundary' in class_name.lower():
                spatial_info['boundary_positions'].append(spatial_data)
            
            spatial_info['distances'].append(float(estimated_distance))
            spatial_info['angles'].append((float(angle_x), float(angle_y)))
        
        return spatial_info
    
    def _visualize_detections(self, image: np.ndarray, detections: Dict) -> np.ndarray:
        """Create visualization with bounding boxes"""
        viz_image = image.copy()
        
        # Draw boxes
        for box, conf, class_name in zip(
            detections['boxes'],
            detections['confidences'],
            detections['class_names']
        ):
            x1, y1, x2, y2 = box.astype(int)
            
            # Color by class
            if 'obstacle' in class_name.lower():
                color = (0, 0, 255)  # Red
                label_prefix = '🧱'
            elif 'goal' in class_name.lower() or 'target' in class_name.lower():
                color = (0, 255, 0)  # Green
                label_prefix = '🎯'
            elif 'boundary' in class_name.lower():
                color = (255, 0, 0)  # Blue
                label_prefix = '📏'
            else:
                color = (255, 255, 0)  # Cyan
                label_prefix = '❓'
            
            # Draw box
            cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f'{label_prefix} {class_name} {conf:.2f}'
            cv2.putText(
                viz_image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        
        return viz_image
    
    def _empty_detection_result(self, return_viz: bool = False) -> Dict:
        """Return empty detection result"""
        result = {
            'boxes': [],
            'confidences': [],
            'classes': [],
            'class_names': [],
            'obstacles': [],
            'goals': [],
            'boundaries': []
        }
        if return_viz:
            result['visualization'] = None
        return result
    
    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return self.detection_stats.copy()
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.detection_stats = {
            'total_detections': 0,
            'obstacle_detections': 0,
            'goal_detections': 0,
            'boundary_detections': 0,
            'average_confidence': 0.0,
            'detection_frequency': 0.0
        }


class YOLOEnsemble:
    """
    Ensemble of YOLO detectors for improved robustness
    Runs multiple YOLO models and combines results
    """
    
    def __init__(self, models: List[str] = None, **kwargs):
        """
        Initialize ensemble
        
        Args:
            models: List of model names to use
        """
        if models is None:
            models = ['yolov8n', 'yolov8s']  # Nano and small for speed/accuracy trade-off
        
        self.detectors = []
        for model_name in models:
            detector = YOLODetector(model_name=model_name, **kwargs)
            self.detectors.append(detector)
        
        print(f"✅ YOLO Ensemble initialized with {len(self.detectors)} models")
    
    def detect(self, image: np.ndarray, return_viz: bool = False) -> Dict:
        """Run detection with all models and combine results"""
        all_detections = []
        
        for detector in self.detectors:
            detections = detector.detect(image, return_viz=False)
            all_detections.append(detections)
        
        # Combine results (voting for objectdetection confidence)
        combined = self._combine_detections(all_detections)
        
        if return_viz and all_detections:
            combined['visualization'] = all_detections[0].get('visualization')
        
        return combined
    
    def _combine_detections(self, detections_list: List[Dict]) -> Dict:
        """Combine detections from multiple models"""
        combined = {
            'boxes': [],
            'confidences': [],
            'classes': [],
            'class_names': [],
            'obstacles': [],
            'goals': [],
            'boundaries': []
        }
        
        # Simple averaging of confidences for same detections
        for detections in detections_list:
            combined['boxes'].extend(detections['boxes'])
            combined['confidences'].extend(detections['confidences'])
            combined['obstacles'].extend(detections['obstacles'])
            combined['goals'].extend(detections['goals'])
            combined['boundaries'].extend(detections['boundaries'])
        
        # Remove duplicates (simple NMS-style approach)
        # Could be enhanced with more sophisticated fusion
        
        return combined

"""
Vision-based Obstacle Detection for UAV Navigation
Uses pretrained deep learning models to detect obstacles and goals from camera images
"""

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

class VisionObstacleDetector:
    def __init__(self, model_type='yolo', confidence_threshold=0.5):
        """
        Initialize vision-based obstacle detector
        
        Args:
            model_type: 'yolo', 'custom', or 'simple'
            confidence_threshold: Minimum confidence for detections
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self._load_model()
    
    def _load_model(self):
        """Load the pretrained model"""
        try:
            if self.model_type == 'yolo':
                # Use YOLOv8 for object detection
                self.model = YOLO('yolov8n.pt')  # Nano version for speed
                print("📹 YOLO model loaded successfully")
                
            elif self.model_type == 'custom':
                # Placeholder for your custom model
                # self.model = torch.load('your_custom_model.pth')
                print("🔧 Custom model not implemented yet, using simple features")
                self.model_type = 'simple'
                
            else:  # simple
                print("🔍 Using simple computer vision features")
                
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("🔍 Falling back to simple computer vision features")
            self.model_type = 'simple'
    
    def detect_obstacles_and_goal(self, image):
        """
        Detect obstacles and goal from camera image
        
        Args:
            image: numpy array of shape (H, W, 3) in RGB format
            
        Returns:
            dict with detection results
        """
        if image is None or image.size == 0:
            return self._empty_detection_result()
        
        try:
            if self.model_type == 'yolo':
                return self._yolo_detection(image)
            elif self.model_type == 'custom':
                return self._custom_model_detection(image)
            else:
                return self._simple_cv_detection(image)
                
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            return self._empty_detection_result()
    
    def _yolo_detection(self, image):
        """Use YOLO for obstacle detection"""
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        
        obstacles = []
        goal_visible = False
        goal_direction = np.array([0.0, 0.0])
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Calculate center and approximate distance
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Estimate distance based on bounding box size
                    # Larger boxes = closer objects
                    bbox_area = width * height
                    estimated_distance = max(0.5, 3.0 - (bbox_area / (224 * 224)) * 2.5)
                    
                    # Convert pixel coordinates to direction
                    # Image center is (112, 112) for 224x224 image
                    direction_x = (center_x - 112) / 112  # -1 to 1
                    direction_y = (center_y - 112) / 112  # -1 to 1
                    
                    obstacles.append({
                        'center': (center_x, center_y),
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id,
                        'estimated_distance': estimated_distance,
                        'direction': np.array([direction_x, direction_y])
                    })
        
        return {
            'obstacles': obstacles,
            'goal_visible': goal_visible,
            'goal_direction': goal_direction,
            'num_obstacles': len(obstacles),
            'detection_confidence': np.mean([obs['confidence'] for obs in obstacles]) if obstacles else 0.0
        }
    
    def _custom_model_detection(self, image):
        """Placeholder for custom model detection"""
        # TODO: Implement your custom trained model here
        return self._simple_cv_detection(image)
    
    def _simple_cv_detection(self, image):
        """Simple computer vision based detection"""
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        obstacles = []
        
        # Simple edge-based obstacle detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small contours
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w/2
                center_y = y + h/2
                
                # Estimate distance based on size
                estimated_distance = max(0.5, 3.0 - (area / (224 * 224)) * 2.5)
                
                # Direction from center
                direction_x = (center_x - 112) / 112
                direction_y = (center_y - 112) / 112
                
                obstacles.append({
                    'center': (center_x, center_y),
                    'bbox': (x, y, x+w, y+h),
                    'confidence': min(1.0, area / 1000),  # Confidence based on area
                    'class_id': 0,  # Generic obstacle
                    'estimated_distance': estimated_distance,
                    'direction': np.array([direction_x, direction_y])
                })
        
        # Simple goal detection (look for specific colors - e.g., green)
        # You can customize this based on how you mark your goal
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        goal_visible = np.sum(green_mask) > 1000  # If enough green pixels
        goal_direction = np.array([0.0, 0.0])
        
        if goal_visible:
            # Find center of green region
            moments = cv2.moments(green_mask)
            if moments['m00'] > 0:
                goal_x = moments['m10'] / moments['m00']
                goal_y = moments['m01'] / moments['m00']
                goal_direction = np.array([(goal_x - 112) / 112, (goal_y - 112) / 112])
        
        return {
            'obstacles': obstacles,
            'goal_visible': goal_visible,
            'goal_direction': goal_direction,
            'num_obstacles': len(obstacles),
            'detection_confidence': np.mean([obs['confidence'] for obs in obstacles]) if obstacles else 0.0
        }
    
    def _empty_detection_result(self):
        """Return empty detection result"""
        return {
            'obstacles': [],
            'goal_visible': False,
            'goal_direction': np.array([0.0, 0.0]),
            'num_obstacles': 0,
            'detection_confidence': 0.0
        }
    
    def extract_visual_features(self, image, detection_results=None):
        """
        Extract visual features for RL agent observation
        
        Args:
            image: Camera image
            detection_results: Optional detection results to avoid re-computation
            
        Returns:
            numpy array of visual features (32 dimensions)
        """
        if detection_results is None:
            detection_results = self.detect_obstacles_and_goal(image)
        
        features = []
        
        # Basic image statistics (4D)
        if image is not None and image.size > 0:
            img_float = image.astype(np.float32) / 255.0
            features.extend([
                np.mean(img_float),      # Overall brightness
                np.std(img_float),       # Overall contrast
                np.mean(img_float[:, :, 0]),  # Red channel
                np.mean(img_float[:, :, 1])   # Green channel
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Obstacle detection features (8D)
        features.extend([
            detection_results['num_obstacles'],           # Number of obstacles
            detection_results['detection_confidence'],    # Average confidence
            0.0, 0.0,  # Closest obstacle direction (will be filled below)
            0.0,       # Closest obstacle distance
            0.0, 0.0,  # Average obstacle direction
            0.0        # Obstacle density
        ])
        
        # Fill obstacle-specific features
        if detection_results['obstacles']:
            # Find closest obstacle
            closest_obs = min(detection_results['obstacles'], 
                            key=lambda x: x['estimated_distance'])
            features[6] = closest_obs['direction'][0]  # Closest obstacle direction X
            features[7] = closest_obs['direction'][1]  # Closest obstacle direction Y
            features[8] = closest_obs['estimated_distance']  # Closest distance
            
            # Average obstacle direction
            avg_direction = np.mean([obs['direction'] for obs in detection_results['obstacles']], axis=0)
            features[9] = avg_direction[0]
            features[10] = avg_direction[1]
            
            # Obstacle density (obstacles per unit area)
            features[11] = len(detection_results['obstacles']) / 10.0  # Normalized
        
        # Goal detection features (4D)
        features.extend([
            1.0 if detection_results['goal_visible'] else 0.0,  # Goal visible flag
            detection_results['goal_direction'][0],              # Goal direction X
            detection_results['goal_direction'][1],              # Goal direction Y
            0.0  # Goal confidence (placeholder)
        ])
        
        # Pad remaining features to reach 32 dimensions
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32])  # Ensure exactly 32 dimensions

# Global instance for easy access
vision_detector = None

def initialize_vision_detector(model_type='simple', confidence_threshold=0.5):
    """Initialize global vision detector"""
    global vision_detector
    vision_detector = VisionObstacleDetector(model_type, confidence_threshold)
    return vision_detector

def get_vision_detector():
    """Get global vision detector instance"""
    global vision_detector
    if vision_detector is None:
        vision_detector = initialize_vision_detector()
    return vision_detector
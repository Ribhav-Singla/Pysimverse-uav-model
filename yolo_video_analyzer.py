"""
YOLO Video Analysis Module
Records and analyzes UAV navigation videos with YOLO object detection
Provides visualization, logging, and performance metrics
"""

import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


class YOLOVideoAnalyzer:
    """
    Records and analyzes UAV navigation videos with YOLO detections
    
    Features:
    - Real-time video recording with YOLO overlay
    - Detection statistics and logging
    - Performance metrics extraction
    - Video file management
    """
    
    def __init__(
        self,
        output_dir: str = "yolo_videos",
        video_name: Optional[str] = None,
        fps: int = 30,
        frame_size: Tuple = (1280, 720),
        enable_yolo_viz: bool = True
    ):
        """
        Initialize video analyzer
        
        Args:
            output_dir: Directory to save videos
            video_name: Custom video name (auto-generated if None)
            fps: Frames per second
            frame_size: Video frame size (width, height)
            enable_yolo_viz: Whether to draw YOLO detections
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.frame_size = frame_size
        self.enable_yolo_viz = enable_yolo_viz
        
        # Generate video filename
        if video_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_name = f"uav_navigation_{timestamp}"
        else:
            self.video_name = video_name
        
        # Initialize video writer
        self.video_path = self.output_dir / f"{self.video_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            self.fps,
            self.frame_size
        )
        
        # Tracking data
        self.frames_written = 0
        self.detection_log = []
        self.frame_metrics = []
        
        print(f"✅ Video analyzer initialized: {self.video_path}")
    
    def write_frame(
        self,
        image: np.ndarray,
        detections: Optional[Dict] = None,
        state_info: Optional[Dict] = None,
        overlay_text: Optional[str] = None
    ):
        """
        Write frame to video with YOLO overlay and state information
        
        Args:
            image: Frame image (will be resized if needed)
            detections: YOLO detections dictionary
            state_info: UAV state information to display
            overlay_text: Additional text to overlay
        """
        # Resize frame to target size
        frame = self._resize_frame(image)
        
        # Draw YOLO detections
        if self.enable_yolo_viz and detections is not None:
            frame = self._draw_detections(frame, detections)
        
        # Draw state information
        if state_info is not None:
            frame = self._draw_state_info(frame, state_info)
        
        # Draw additional text
        if overlay_text is not None:
            frame = self._draw_text(frame, overlay_text)
        
        # Write frame
        self.writer.write(frame)
        self.frames_written += 1
        
        # Log detection data
        if detections is not None:
            self.detection_log.append({
                'frame': self.frames_written,
                'detections': detections,
                'timestamp': datetime.now().isoformat()
            })
        
        # Log state metrics
        if state_info is not None:
            self.frame_metrics.append({
                'frame': self.frames_written,
                'state': state_info,
                'timestamp': datetime.now().isoformat()
            })
    
    def _resize_frame(self, image: np.ndarray) -> np.ndarray:
        """Resize frame to target size"""
        if image.shape[:2][::-1] != self.frame_size:
            image = cv2.resize(image, self.frame_size)
        return image
    
    def _draw_detections(self, frame: np.ndarray, detections: Dict) -> np.ndarray:
        """Draw YOLO detections on frame"""
        if not detections.get('boxes'):
            return frame
        
        for box, conf, class_name in zip(
            detections.get('boxes', []),
            detections.get('confidences', []),
            detections.get('class_names', [])
        ):
            x1, y1, x2, y2 = box.astype(int)
            
            # Scale box coordinates to frame size
            scale_x = self.frame_size[0] / (detections.get('_image_width', 640))
            scale_y = self.frame_size[1] / (detections.get('_image_height', 480))
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            # Color by class
            if 'obstacle' in class_name.lower():
                color = (0, 0, 255)  # Red
            elif 'goal' in class_name.lower() or 'target' in class_name.lower():
                color = (0, 255, 0)  # Green
            elif 'boundary' in class_name.lower():
                color = (255, 0, 0)  # Blue
            else:
                color = (255, 255, 0)  # Cyan
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f'{class_name} {conf:.2f}'
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - baseline),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                frame, label, (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        return frame
    
    def _draw_state_info(self, frame: np.ndarray, state_info: Dict) -> np.ndarray:
        """Draw UAV state information on frame"""
        y_offset = 30
        line_height = 25
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw state text
        text_color = (0, 255, 0)
        
        if 'position' in state_info:
            pos = state_info['position']
            text = f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            y_offset += line_height
        
        if 'velocity' in state_info:
            vel = state_info['velocity']
            speed = np.linalg.norm(vel)
            text = f"Speed: {speed:.2f} m/s"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            y_offset += line_height
        
        if 'goal_distance' in state_info:
            goal_dist = state_info['goal_distance']
            text = f"Goal Dist: {goal_dist:.2f} m"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            y_offset += line_height
        
        if 'collision' in state_info:
            status = "COLLISION!" if state_info['collision'] else "Safe"
            color = (0, 0, 255) if state_info['collision'] else (0, 255, 0)
            text = f"Status: {status}"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += line_height
        
        if 'episode' in state_info:
            text = f"Episode: {state_info['episode']}"
            cv2.putText(frame, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            y_offset += line_height
        
        return frame
    
    def _draw_text(self, frame: np.ndarray, text: str) -> np.ndarray:
        """Draw custom text on frame"""
        cv2.putText(
            frame, text, (20, self.frame_size[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        return frame
    
    def finalize(self) -> Dict:
        """
        Finalize video recording and generate statistics
        
        Returns:
            Summary dictionary with statistics
        """
        self.writer.release()
        
        # Generate statistics
        stats = self._generate_statistics()
        
        # Save statistics
        self._save_statistics(stats)
        
        print(f"✅ Video saved: {self.video_path}")
        print(f"   Frames: {self.frames_written}")
        print(f"   Duration: {self.frames_written / self.fps:.2f}s")
        print(f"   Total Detections: {len(self.detection_log)}")
        
        return stats
    
    def _generate_statistics(self) -> Dict:
        """Generate statistics from logged data"""
        stats = {
            'video_path': str(self.video_path),
            'total_frames': self.frames_written,
            'duration_seconds': self.frames_written / self.fps,
            'fps': self.fps,
            'detection_frames': len(self.detection_log),
            'state_frames': len(self.frame_metrics),
        }
        
        # Aggregate detection statistics
        if self.detection_log:
            obstacle_count = sum(
                len(det['detections'].get('obstacles', []))
                for det in self.detection_log
            )
            goal_count = sum(
                len(det['detections'].get('goals', []))
                for det in self.detection_log
            )
            boundary_count = sum(
                len(det['detections'].get('boundaries', []))
                for det in self.detection_log
            )
            
            stats['total_obstacles_detected'] = obstacle_count
            stats['total_goals_detected'] = goal_count
            stats['total_boundaries_detected'] = boundary_count
        
        # Aggregate state statistics
        if self.frame_metrics:
            positions = np.array([
                m['state'].get('position', [0, 0, 0])
                for m in self.frame_metrics
                if 'position' in m['state']
            ])
            
            if len(positions) > 0:
                stats['trajectory_distance'] = np.sum([
                    np.linalg.norm(positions[i+1] - positions[i])
                    for i in range(len(positions)-1)
                ])
        
        return stats
    
    def _save_statistics(self, stats: Dict):
        """Save statistics to log file"""
        log_path = self.output_dir / f"{self.video_name}_stats.txt"
        
        with open(log_path, 'w') as f:
            f.write("=== YOLO Video Analysis Statistics ===\n\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"✅ Statistics saved: {log_path}")
    
    def get_statistics(self) -> Dict:
        """Get current statistics (before finalization)"""
        return {
            'frames_written': self.frames_written,
            'detections_logged': len(self.detection_log),
            'state_frames_logged': len(self.frame_metrics)
        }


class VideoFrameBuffer:
    """
    Efficient frame buffer for video recording
    Allows batching frames for better performance
    """
    
    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.frames = []
        self.metadata = []
    
    def add_frame(self, frame: np.ndarray, metadata: Optional[Dict] = None):
        """Add frame to buffer"""
        self.frames.append(frame.copy())
        self.metadata.append(metadata or {})
        
        # Flush if buffer is full
        if len(self.frames) >= self.buffer_size:
            self.flush()
    
    def flush(self) -> Tuple[list, list]:
        """Flush buffer and return frames and metadata"""
        frames = self.frames.copy()
        metadata = self.metadata.copy()
        
        self.frames = []
        self.metadata = []
        
        return frames, metadata
    
    def get_pending_count(self) -> int:
        """Get number of pending frames in buffer"""
        return len(self.frames)

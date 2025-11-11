"""
YOLO Integration Setup Guide and Utilities
Complete installation and configuration for YOLO-based UAV perception
"""

import subprocess
import sys
from pathlib import Path


class YOLOSetup:
    """
    Setup and configuration utilities for YOLO integration
    """
    
    @staticmethod
    def install_dependencies():
        """Install required packages for YOLO integration"""
        packages = [
            'ultralytics>=8.0.0',  # YOLOv8
            'opencv-python>=4.8.0',  # Computer vision
            'torch>=2.0.0',  # PyTorch
            'numpy>=1.24.0',
        ]
        
        print("🚀 Installing YOLO dependencies...")
        for package in packages:
            print(f"   Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        
        print("✅ All dependencies installed!")
    
    @staticmethod
    def download_yolo_models():
        """Download pretrained YOLO models"""
        print("📥 Downloading YOLO models...")
        try:
            from ultralytics import YOLO
            
            models = ['yolov8n.pt', 'yolov8s.pt']  # nano and small
            for model in models:
                print(f"   Downloading {model}...")
                yolo = YOLO(model)
                print(f"   ✅ {model} ready")
        except ImportError:
            print("❌ YOLOv8 not available. Please run: pip install ultralytics")
    
    @staticmethod
    def get_installation_instructions():
        """Print installation instructions"""
        instructions = """
╔════════════════════════════════════════════════════════════════╗
║         YOLO Integration Installation Guide                   ║
╚════════════════════════════════════════════════════════════════╝

Step 1: Install Required Packages
──────────────────────────────────
pip install ultralytics opencv-python torch torchvision

Step 2: Download YOLO Models (Optional - auto-downloads on first use)
──────────────────────────────────────────────────────────────────
python -c "from yolo_setup import YOLOSetup; YOLOSetup.download_yolo_models()"

Step 3: Configure UAV Environment
──────────────────────────────────
In your training script, add:

```python
from yolo_detector import YOLODetector
from yolo_cnn_fusion import CNNYOLOFusion, FusionObservationBuilder
from yolo_video_analyzer import YOLOVideoAnalyzer

# Initialize YOLO detector
yolo_detector = YOLODetector(model_name='yolov8n', device='cuda')

# Create fusion module
fusion_module = CNNYOLOFusion(
    depth_cnn=env.depth_extractor,
    yolo_detector=yolo_detector,
    fusion_mode='concatenate'
)

# Build observations with fusion
obs_builder = FusionObservationBuilder(fusion_module)

# For video recording
video_recorder = YOLOVideoAnalyzer(
    output_dir='yolo_videos',
    fps=30,
    enable_yolo_viz=True
)
```

Step 4: Training Integration
────────────────────────────
The fusion module integrates seamlessly with existing PPO training:

```python
# In training loop:
obs, info = env.reset()
rgb_image = env.render_rgb()  # Get RGB from environment
depth_image = env.render_depth()  # Get depth from camera

# Build fused observation
fused_obs = obs_builder.build(
    pos=obs[:3],
    vel=obs[3:6],
    goal_pos=env.goal_pos,
    depth_image=depth_image,
    rgb_image=rgb_image
)

# Use fused observation for policy
action = agent.select_action(fused_obs)

# Optionally record video
if recording:
    video_recorder.write_frame(
        rgb_image,
        detections=yolo_detector.detect(rgb_image),
        state_info={
            'position': obs[:3],
            'velocity': obs[3:6],
            'goal_distance': np.linalg.norm(env.goal_pos - obs[:3])
        }
    )
```

Step 5: Feature Dimensions
──────────────────────────
Observation space breakdown (160D):
  - Position: 3D
  - Velocity: 3D
  - Goal distance: 3D
  - CNN depth features: 128D
  - YOLO detection features: 18D
  - Navigation features: 5D
  ─────────────────────────
  Total: 160D

Step 6: Performance Considerations
──────────────────────────────────
- YOLOv8n (nano): Fastest, ~5-10ms inference
- YOLOv8s (small): Good balance, ~10-20ms inference
- YOLOv8m (medium): Better accuracy, ~20-40ms inference

For real-time performance, use nano or small models.
Enable GPU acceleration if available (CUDA).

Step 7: Troubleshooting
──────────────────────
Issue: YOLO model download timeout
Solution: Download manually or use offline mode

Issue: Out of memory
Solution: Use smaller model (nano) or reduce image resolution

Issue: Slow inference
Solution: Use GPU acceleration or smaller model

For more info: https://github.com/ultralytics/ultralytics
"""
        print(instructions)


YOLO_INTEGRATION_CHECKLIST = """
═══════════════════════════════════════════════════════════════════
YOLO Integration Checklist for UAV Navigation System
═══════════════════════════════════════════════════════════════════

Pre-Installation
─────────────────
□ Python 3.8+ installed
□ pip updated (pip install --upgrade pip)
□ CUDA toolkit available (optional, for GPU acceleration)

Installation
─────────────
□ Install ultralytics: pip install ultralytics
□ Install OpenCV: pip install opencv-python
□ Install PyTorch: pip install torch torchvision
□ Verify installation: python -c "from ultralytics import YOLO"

Module Setup
─────────────
□ yolo_detector.py - Created and configured
□ yolo_cnn_fusion.py - Created and configured
□ yolo_video_analyzer.py - Created and configured
□ yolo_setup.py - Setup utilities ready

Environment Integration
────────────────────────
□ YOLODetector initialized in training
□ CNNYOLOFusion module created
□ FusionObservationBuilder instantiated
□ Observation space extended to 160D

Training Integration
─────────────────────
□ Depth images available from MuJoCo
□ RGB images available from environment
□ Fused observations passed to PPO agent
□ Video recording enabled (optional)

Validation
──────────
□ YOLO can detect objects in test images
□ Fusion produces 160D observations
□ Training runs without errors
□ Video saves correctly (if enabled)

Performance Optimization
────────────────────────
□ GPU enabled if available
□ Batch processing considered
□ Model caching enabled
□ Frame buffering implemented (optional)

Documentation
───────────────
□ Detection format documented
□ Feature dimensions clear
□ Configuration options explained
□ Performance benchmarks recorded

═══════════════════════════════════════════════════════════════════
"""


if __name__ == '__main__':
    print(YOLO_INTEGRATION_CHECKLIST)
    YOLOSetup.get_installation_instructions()

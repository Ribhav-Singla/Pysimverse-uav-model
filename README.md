# UAV Navigation with PPO and Vision-Based Detection

A reinforcement learning project that trains an Unmanned Aerial Vehicle (UAV) to navigate through obstacle-rich environments using Proximal Policy Optimization (PPO) and camera-based vision detection with deep learning models.

## 🚁 Project Overview

This project implements an intelligent UAV navigation system using:
- **MuJoCo Physics Engine** for realistic 3D simulation
- **PPO (Proximal Policy Optimization)** for reinforcement learning
- **Vision-Based Detection** using YOLOv8 and computer vision models
- **Camera Rendering System** with adaptive FPS control (0.5-10.0 FPS)
- **Custom Gymnasium Environment** with dynamic obstacles and boundary enforcement
- **Real-time Visualization** with trajectory tracking
- **Neurosymbolic Integration** with rule-based navigation guidance

## 🎯 Features

- **Vision-Based Navigation**: Camera rendering with YOLOv8 object detection
- **Adaptive FPS Control**: Configurable camera update rates (0.5-10.0 FPS) for performance optimization
- **Smart Obstacle Detection**: 32D visual features including object classification, distance estimation, and spatial awareness
- **Neurosymbolic System**: Rule-based navigation guidance with 11 comprehensive navigation rules
- **Dynamic Environment**: Static and dynamic obstacles with configurable density
- **Collision Detection**: Precise collision boundaries (0.2m threshold)
- **Reward Engineering**: Vision-based proximity penalties and goal incentives
- **Adaptive Training**: Action standard deviation decay and learning rate scheduling
- **Real-time Visualization**: Trajectory tracking with camera view rendering
- **Model Persistence**: Separate weight files for standard and neurosymbolic training

## 📁 Project Structure

```
pysimverse/
├── uav_env.py                    # Custom Gymnasium environment with vision system
├── ppo_agent.py                  # PPO agent with actor-critic networks
├── training.py                   # Standard PPO training loop
├── neurosymbolic_training.py     # Neurosymbolic rule-guided training
├── neurosymbolic_ppo_agent.py    # Enhanced PPO agent with rule integration
├── neurosymbolic_rdr.py          # Ripple Down Rules knowledge base
├── vision_obstacle_detector.py   # Vision-based detection system
├── uav_render.py                 # Standard visualization and rendering
├── uav_render_neurosymbolic.py   # Neurosymbolic model visualization
├── environment.xml               # MuJoCo environment configuration
├── uav_model.xml                 # UAV physics model definition
├── uav_navigation_rules.json     # Navigation rules knowledge base
├── yolov8n.pt                    # YOLOv8 pretrained model
├── PPO_preTrained/               # Saved model weights
│   ├── PPO_UAV_Weights.pth           # Standard training weights
│   └── PPO_UAV_Weights_neurosymbolic.pth # Neurosymbolic weights
└── pysim_env/                    # Python virtual environment
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- MuJoCo physics engine
- CUDA (optional, for GPU acceleration)

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ribhav-Singla/Pysimverse-uav-model.git
   cd Pysimverse-uav-model
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv pysim_env
   # On Windows:
   pysim_env\Scripts\activate
   # On Linux/Mac:
   source pysim_env/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision
   pip install mujoco
   pip install gymnasium
   pip install numpy matplotlib opencv-python
   pip install PyOpenGL glfw imageio
   pip install ultralytics  # For YOLOv8
   pip install pillow      # For image processing
   ```

## 🚀 Usage

### Training the Model
```bash
# Activate virtual environment
pysim_env\Scripts\activate

# Standard PPO training with vision system
python training.py

# Neurosymbolic rule-guided training
python neurosymbolic_training.py
```

### Rendering and Visualization
```bash
# Standard model visualization
python uav_render.py

# Neurosymbolic model with rule explanations
python uav_render_neurosymbolic.py
```

### Camera System Configuration
```bash
# Test vision detection system
python -c "from vision_obstacle_detector import VisionObstacleDetector; print('Vision system ready!')"

# Adjust camera FPS in uav_env.py CONFIG:
CONFIG['camera_fps'] = 2.0  # 2 FPS (updates every 10 steps)
CONFIG['camera_fps'] = 10.0 # 10 FPS (updates every step)
```

## 🧠 Model Architecture

### Actor-Critic Networks
- **Input**: 41-dimensional state space (position, velocity, visual features)
- **Actor Network**: 3 hidden layers (256 units each) with LayerNorm
- **Critic Network**: 3 hidden layers (256 units each) with LayerNorm
- **Output**: 3-dimensional continuous action space (3D velocity)

### State Space (41 dimensions)
- **Position**: [x, y, z] coordinates (3D)
- **Velocity**: [vx, vy, vz] components (3D)
- **Goal Distance**: [dx, dy, dz] to target (3D)
- **Visual Features**: Camera-based detection features (32D)
  - Image statistics (brightness, contrast, color channels)
  - Obstacle detection (count, confidence, direction, distance)
  - Goal detection (visibility, direction, confidence)
  - Spatial awareness (clearance zones, navigation confidence)

### Action Space (3 dimensions)
- **Linear Velocity**: [vx, vy, vz] changes
- **Constraints**: Velocity range [0.15, 0.5] m/s
- **Note**: Constant altitude maintained automatically

## 🎛️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gamma` | 0.999 | Discount factor for long-term planning |
| `update_timestep` | 2048 | Policy update frequency |
| `K_epochs` | 10 | PPO epochs per update |
| `eps_clip` | 0.1 | PPO clipping parameter |
| `lr_actor` | 0.0001 | Actor learning rate |
| `lr_critic` | 0.0005 | Critic learning rate |
| `action_std` | 0.3→0.1 | Adaptive exploration decay |

## 🏆 Reward System

### Reward Components
- **Goal Progress**: +10.0 × (progress toward goal)
- **Goal Achievement**: +1000 for reaching target (< 0.5m) + stabilization requirement
- **Vision-Based Proximity Penalties**: 
  - Very close obstacles (< 0.3m): -5.0
  - Close obstacles (< 0.5m): -2.0
  - Caution zone (< 1.0m): -0.5
- **Safe Navigation Bonus**: +0.5 for maintaining safe distance while progressing
- **Directional Bias**: Rewards eastward movement when goal is eastward
- **Boundary Penalties**: -10.0 × proximity to boundaries
- **Collision Penalty**: -100 for obstacle collision
- **Out-of-Bounds Penalty**: -100 for leaving valid area

### Termination Conditions
- Collision with obstacles (0.2m threshold)
- Boundary violation
- Goal achievement with stabilization (< 0.5m for 10 consecutive steps)
- Maximum episode length (50,000 steps)

## 📊 Performance Monitoring

The training script provides real-time monitoring:
- **Episode Length**: Steps per episode
- **Episode Reward**: Cumulative reward
- **Action Standard Deviation**: Exploration decay
- **Best Model Tracking**: Automatic best model saving

Example output:
```
Episode 1    Length: 45     Reward: -89.5
Episode 2    Length: 67     Reward: -72.3
Episode 3    Length: 123    Reward: -45.7
✓ New best model saved with reward: -45.7
```

## 🔧 Configuration

### Environment Configuration (`uav_env.py`)
```python
CONFIG = {
    'world_size': 8.0,               # Environment boundaries (8x8m)
    'start_pos': [-3, -3, 1.0],      # UAV starting position
    'goal_pos': [3, 3, 1.0],         # Target position (dynamic)
    'uav_flight_height': 1.0,        # Constant flight altitude
    
    # Vision System
    'camera_enabled': True,          # Enable camera rendering
    'camera_fps': 2.0,              # Camera update frequency
    'camera_width': 224,            # Image width
    'camera_height': 224,           # Image height
    'vision_detection_model': 'yolo', # Detection model type
    
    # Physics
    'control_dt': 0.02,             # Control timestep (50 Hz)
    'max_velocity': 2.0,            # Maximum UAV velocity
}
```

### Training Configuration (`training.py`)
- Modify hyperparameters in the `main()` function
- Adjust network architecture in `ppo_agent.py`
- Configure environment parameters in `uav_env.py`

## 📹 Vision System Architecture

### Camera Rendering Pipeline
1. **MuJoCo Integration**: UAV-mounted camera renders 224×224 RGB images
2. **Adaptive FPS Control**: Configurable update rates (0.5-10.0 FPS) for performance optimization
3. **Real-time Processing**: Vision features cached between updates to maintain smooth simulation

### Obstacle Detection Methods

#### YOLOv8 Detection (Primary)
- **Model**: YOLOv8n (6.2MB nano model) for real-time performance
- **Object Classification**: Detects and classifies objects in camera view
- **Distance Estimation**: `distance = 3.0 - (bbox_area/total_area) × 2.5`
- **Direction Calculation**: `direction = (center - image_center) / image_center`

#### Computer Vision Fallback
- **Edge Detection**: Canny filters for object boundaries
- **Color Analysis**: HSV space obstacle detection
- **Contour Analysis**: Geometric shape recognition

### Visual Feature Extraction (32D)
```python
# Feature categories:
- Basic Image Stats (4D):    Brightness, contrast, color channels
- Obstacle Features (8D):    Count, confidence, direction, distance, density
- Goal Detection (4D):       Visibility, direction, distance, confidence
- Spatial Awareness (8D):    Clearance zones, navigation confidence
- Movement Analysis (8D):    Optical flow, trajectory validation
```

### Performance Comparison: Vision vs LIDAR

| Aspect | Vision System ✅ | LIDAR System ❌ |
|--------|------------------|------------------|
| **Realism** | Real-world applicable | Synthetic ray-casting |
| **Information** | Rich visual context (color, texture, shape) | Distance-only measurements |
| **Object Recognition** | Classifies object types | Unknown object types |
| **Computational Cost** | Efficient with FPS control | CPU-intensive ray-casting |
| **Transfer Learning** | Applicable to real drones | Simulation-only solution |

## 📈 Results and Analysis

The vision-based system demonstrates significant improvements over LIDAR:

### Performance Metrics
- **Observation Space**: Optimized from 68D (hybrid) to 41D (vision-only)
- **Computational Efficiency**: Vision updates at 2.0 FPS vs LIDAR every step
- **Detection Accuracy**: Object classification vs distance-only measurements
- **Real-world Applicability**: Camera-based system directly transferable to real UAVs

### Training Results
- **Enhanced Spatial Understanding**: 32D visual features provide richer environmental context
- **Smarter Obstacle Avoidance**: Object type recognition enables appropriate responses
- **Improved Goal Navigation**: Visual goal detection with confidence scoring
- **Stable Learning**: Neurosymbolic rules provide guidance during exploration

### Neurosymbolic Integration
- **Rule-Based Guidance**: 11 comprehensive navigation rules with continuous actions
- **Human-Interpretable Decisions**: Rule firing provides explainable AI behavior
- **Adaptive Learning**: Rules complement RL exploration while maintaining safety

## 🐛 Troubleshooting

### Common Issues
1. **MuJoCo Installation**: Ensure MuJoCo is properly installed and licensed
2. **YOLO Model Download**: YOLOv8n.pt downloads automatically on first run
3. **Camera Rendering**: Ensure graphics drivers support OpenGL
4. **GPU Memory**: Reduce batch size if CUDA out of memory errors occur
5. **Vision FPS**: Lower camera_fps if experiencing performance issues
6. **Model Loading**: Architecture mismatch (41D vs 68D) requires training from scratch

### Debug Mode
```python
# Enable detailed logging in training.py
render = True  # Enable visualization
log_interval = 1  # Log every episode

# Vision system debugging in uav_env.py
CONFIG['camera_fps'] = 10.0  # Max FPS for debugging
print(f"Vision features shape: {visual_features.shape}")  # Check feature dimensions
```

### Performance Optimization
```python
# Optimize camera FPS for your hardware:
CONFIG['camera_fps'] = 0.5   # Slow hardware: 0.5 FPS (every 40 steps)
CONFIG['camera_fps'] = 2.0   # Balanced: 2.0 FPS (every 10 steps) 
CONFIG['camera_fps'] = 10.0  # Fast hardware: 10.0 FPS (every step)
```

## 📝 Citation

If you use this project in your research, please cite:
```bibtex
@article{uav_vision_navigation,
  title={UAV Navigation with PPO and Vision-Based Detection},
  author={Ribhav Singla},
  year={2025},
  journal={GitHub Repository},
  url={https://github.com/Ribhav-Singla/Pysimverse-uav-model},
  note={Vision-based obstacle detection with YOLOv8 and neurosymbolic integration}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

**Ribhav Singla**
- GitHub: [@Ribhav-Singla](https://github.com/Ribhav-Singla)
- Project Link: [https://github.com/Ribhav-Singla/Pysimverse-uav-model](https://github.com/Ribhav-Singla/Pysimverse-uav-model)

---

⭐ **Star this repository if you found it helpful!**
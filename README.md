# UAV Navigation with PPO and LIDAR

A reinforcement learning project that trains an Unmanned Aerial Vehicle (UAV) to navigate through obstacle-rich environments using Proximal Policy Optimization (PPO) and LIDAR-based obstacle detection.

## 🚁 Project Overview

This project implements an intelligent UAV navigation system using:
- **MuJoCo Physics Engine** for realistic 3D simulation
- **PPO (Proximal Policy Optimization)** for reinforcement learning
- **LIDAR Sensor Simulation** for obstacle detection (16-ray sensor with 3.0m range)
- **Custom Gymnasium Environment** with dynamic obstacles and boundary enforcement
- **Real-time Visualization** with trajectory tracking

## 🎯 Features

- **Dynamic Environment**: Static and dynamic obstacles with configurable density
- **LIDAR Integration**: 16-ray LIDAR sensor for 360° obstacle detection
- **Collision Detection**: Precise collision boundaries (0.2m threshold)
- **Reward Engineering**: Survival bonuses, goal incentives, and penalty system
- **Adaptive Training**: Action standard deviation decay and learning rate scheduling
- **Trajectory Visualization**: Real-time path tracking and rendering
- **Model Persistence**: Automatic saving/loading of trained weights

## 📁 Project Structure

```
pysimverse/
├── uav_env.py          # Custom Gymnasium environment
├── ppo_agent.py        # PPO agent with actor-critic networks
├── training.py         # Training loop and hyperparameter management
├── uav_render.py       # Visualization and rendering
├── environment.xml     # MuJoCo environment configuration
├── uav_model.xml       # UAV physics model definition
├── test_environment.py # Environment testing utilities
├── test_lidar.py       # LIDAR sensor testing
├── PPO_preTrained/     # Saved model weights
└── pysim_env/          # Python virtual environment
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
   ```

## 🚀 Usage

### Training the Model
```bash
# Activate virtual environment
pysim_env\Scripts\activate

# Start training
python training.py
```

### Rendering and Visualization
```bash
# Run with visualization
python uav_render.py
```

### Testing Components
```bash
# Test environment functionality
python test_environment.py

# Test LIDAR sensor
python test_lidar.py
```

## 🧠 Model Architecture

### Actor-Critic Networks
- **Input**: 25-dimensional state space (position, velocity, LIDAR readings)
- **Actor Network**: 3 hidden layers (128 units each) with LayerNorm
- **Critic Network**: 3 hidden layers (128 units each) with LayerNorm
- **Output**: 4-dimensional continuous action space (3D velocity + rotation)

### State Space (25 dimensions)
- **Position**: [x, y, z] coordinates
- **Velocity**: [vx, vy, vz] components
- **LIDAR**: 16-ray distance measurements
- **Goal Information**: Distance and direction to target

### Action Space (4 dimensions)
- **Linear Velocity**: [vx, vy, vz] changes
- **Angular Velocity**: [ωz] yaw rotation
- **Constraints**: Velocity range [0.15, 0.5] m/s

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
- **Survival Bonus**: +0.01 per timestep
- **Goal Progress**: +0.2 for moving toward target
- **Goal Achievement**: +100 for reaching target (< 0.5m)
- **Collision Penalty**: -100 for obstacle collision
- **Boundary Penalty**: -10 for leaving valid area

### Termination Conditions
- Collision with obstacles (0.2m threshold)
- Boundary violation
- Goal achievement (< 0.5m to target)
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
    'boundary_size': [10, 10, 5],     # Environment boundaries
    'obstacle_count': 8,              # Number of obstacles
    'goal_pos': [8, 8, 2],           # Target position
    'lidar_range': 3.0,              # LIDAR maximum range
    'step_reward': 0.01,             # Survival bonus
    'boundary_penalty': -10,         # Out-of-bounds penalty
}
```

### Training Configuration (`training.py`)
- Modify hyperparameters in the `main()` function
- Adjust network architecture in `ppo_agent.py`
- Configure environment parameters in `uav_env.py`

## 📈 Results and Analysis

The optimized model demonstrates:
- **Improved Episode Length**: Survival time increased significantly
- **Efficient Navigation**: Direct paths to goals while avoiding obstacles
- **Robust Collision Avoidance**: Effective LIDAR-based obstacle detection
- **Stable Learning**: Consistent performance improvement over training

## 🐛 Troubleshooting

### Common Issues
1. **MuJoCo Installation**: Ensure MuJoCo is properly installed and licensed
2. **GPU Memory**: Reduce batch size if CUDA out of memory errors occur
3. **Environment Reset**: Check MuJoCo XML files for proper configuration
4. **Model Loading**: Architecture mismatch requires training from scratch

### Debug Mode
```python
# Enable detailed logging in training.py
render = True  # Enable visualization
log_interval = 1  # Log every episode
```

## 📝 Citation

If you use this project in your research, please cite:
```bibtex
@article{uav_ppo_navigation,
  title={UAV Navigation with PPO and LIDAR},
  author={Ribhav Singla},
  year={2025},
  journal={GitHub Repository},
  url={https://github.com/Ribhav-Singla/Pysimverse-uav-model}
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
# UAV Manual Control System

This file provides a manual UAV control simulation that allows users to navigate a UAV from start to goal using arrow keys while recording data for potential training use.

## Features

### Manual Control
- **Arrow Keys**: Control UAV movement in all 4 directions
  - ↑ Arrow Up: Move Forward (+Y direction)  
  - ↓ Arrow Down: Move Backward (-Y direction)
  - ← Arrow Left: Move Left (-X direction)
  - → Arrow Right: Move Right (+X direction)
- **SPACE**: Stop/Brake (set velocity to zero)
- **R**: Reset UAV to start position
- **ESC**: Exit simulation

### Data Recording
- **Automatic Recording**: All flight data is recorded in a buffer
- **S**: Save recorded data to timestamped pickle file
- **C**: Clear recording buffer
- Records: position, velocity, goal distance, LIDAR readings, actions, collisions, goal status

### Environment Features
- Same configuration as the render and training environments
- Random obstacle generation (9 static obstacles)
- Random start position (corner) and goal position (anywhere)
- Real-time collision detection
- Path trail visualization (green trail)
- LIDAR sensor simulation (360° with 16 rays)

## Usage

1. **Installation Requirements**:
   ```bash
   pip install keyboard torch numpy mujoco
   ```

2. **Run the Manual Control System**:
   ```bash
   # Activate virtual environment if using one
   .\venv\Scripts\activate  # Windows
   
   # Run the manual control
   python uav_manual_control.py
   ```

3. **Navigation**:
   - Use arrow keys to navigate from the green start marker to blue goal marker
   - Avoid obstacles (brown/colored objects)
   - Stay within world boundaries
   - UAV automatically maintains flight height

4. **Data Collection**:
   - Flight data is automatically recorded while flying
   - Press 'S' to save data to file (format: `manual_control_data_YYYYMMDD_HHMMSS.pkl`)
   - Use recorded data for analysis or training

## Data Format

The recorded data contains the following information for each timestep:

```python
{
    'timestamp': float,           # System timestamp
    'position': np.array([x,y,z]), # UAV position
    'velocity': np.array([vx,vy,vz]), # UAV velocity
    'goal_position': np.array([gx,gy,gz]), # Goal position
    'goal_distance': float,       # Distance to goal
    'lidar_readings': np.array(16), # LIDAR sensor data (normalized 0-1)
    'action': np.array([ax,ay]),  # Manual control action (velocity commands)
    'collision': bool,            # Whether collision occurred
    'goal_reached': bool          # Whether goal was reached
}
```

## Configuration

The system uses the same configuration as the training environment (`CONFIG` dictionary):

- **World Size**: 8x8 meters
- **Flight Height**: 1.0 meters
- **Manual Speed**: 0.8 m/s base movement speed
- **Control Rate**: 20 Hz (0.05s timestep)
- **LIDAR Range**: 2.9 meters
- **Collision Distance**: 0.1 meters

## Integration with Training

The recorded manual control data is compatible with the PPO training system:

1. **State Space**: 36-dimensional observation space (same as training)
   - Position (3D), Velocity (3D), Goal distance (3D)
   - LIDAR readings (16D), LIDAR features (11D)

2. **Action Space**: 2-dimensional continuous action space
   - X-velocity command, Y-velocity command
   - Z-velocity controlled automatically for stable height

3. **Data Usage**: Recorded data can be used for:
   - Behavioral cloning (imitation learning)
   - Demonstration data for training
   - Performance comparison with AI agents
   - Environment testing and validation

## Troubleshooting

### Common Issues:

1. **Import Errors**: 
   - Make sure all required packages are installed
   - Activate virtual environment if using one

2. **Keyboard Not Responding**:
   - Make sure the simulation window has focus
   - Try running with administrator privileges (Windows)

3. **MuJoCo Viewer Issues**:
   - Ensure MuJoCo is properly installed
   - Check that your system supports OpenGL

4. **Performance Issues**:
   - Reduce `path_trail_length` in CONFIG for lower memory usage
   - Close other applications for better performance

## Files Generated

- `environment.xml`: Temporary MuJoCo environment file
- `manual_control_data_YYYYMMDD_HHMMSS.pkl`: Recorded flight data files

## Comparison with Other Files

- `uav_render.py`: AI agent demonstration (PPO-controlled)
- `uav_manual_control.py`: Manual control with data recording (this file)
- `training.py`: Train PPO agent
- `uav_env.py`: Environment definition for training
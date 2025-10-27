# UAV Performance Comparison System

## Overview

This system compares UAV navigation performance across three approaches:

1. **Human Expert** - Manual keyboard control using the enhanced manual control interface
2. **Neural Only** - Pure PPO agent (lambda=0.0) 
3. **Neurosymbolic** - PPO agent with RDR rules (lambda=1.0)

## System Architecture

### Core Components

1. **`performance_comparison.py`** - Main comparison orchestrator
2. **`uav_agent_runner.py`** - Parameterized agent runner (based on uav_render.py)
3. **`uav_manual_control_wrapper.py`** - Manual control integration with performance tracking
4. **`test_comparison_system.py`** - System verification script

### Execution Flow

```
Level 1 (1 obstacle) → Level 2 (2 obstacles) → ... → Level 10 (10 obstacles)
    │
    ├── Human Expert Trial
    │   └── Manual control with keyboard (Arrow keys, SPACE, R, ESC)
    │
    ├── Neural Only Trial  
    │   └── PPO agent (lambda_0_0.pth) with MuJoCo viewer
    │
    └── Neurosymbolic Trial
        └── PPO + RDR agent (lambda_1_0.pth) with MuJoCo viewer
```

## Quick Start

### 1. Verify System Setup
```bash
python test_comparison_system.py
```

### 2. Run Full Comparison
```bash
python performance_comparison.py
```

### 3. Run Individual Agent Test
```bash
# Neural only
python uav_agent_runner.py --model_path "PPO_preTrained/UAVEnv/PPO_UAV_Weights_lambda_0_0.pth" --lambda_val 0.0

# Neurosymbolic  
python uav_agent_runner.py --model_path "PPO_preTrained/UAVEnv/PPO_UAV_Weights_lambda_1_0.pth" --lambda_val 1.0
```

## Manual Control Interface

### Controls
- **Arrow Keys (↑↓←→)**: UAV movement
- **SPACE**: Stop/brake
- **R**: Reset UAV position
- **ESC**: Exit current trial

### Visual Elements
- **Red UAV**: Your controlled vehicle
- **Green marker**: Start position
- **Blue marker**: Goal position  
- **Colored obstacles**: Navigate around these
- **Green trail**: Path history

## Performance Metrics

For each trial, the system tracks:

- **Success Rate**: Whether the goal was reached
- **Path Length**: Total distance traveled (meters)
- **Step Count**: Number of simulation steps
- **Final Distance**: Distance to goal at trial end
- **Duration**: Time taken (seconds)
- **Path Efficiency**: Direct distance / actual path length
- **Failure Type**: Collision, out-of-bounds, or timeout

## Output

### Console Output
- Real-time trial progress
- Level summaries comparing all approaches
- Final aggregate statistics across all levels

### CSV Export
Results automatically saved to `performance_comparison_YYYYMMDD_HHMMSS.csv` with detailed metrics for analysis.

## Level Progression

- **Level 1**: 1 obstacle
- **Level 2**: 2 obstacles
- **...**
- **Level 10**: 10 obstacles

Each level uses:
- Consistent start position (bottom-left corner)
- Random goal position (one of the other three corners)
- Randomly distributed obstacles with safe positioning
- Same environment for all three approaches

## User Interaction

After each level completion:
```
🔄 Level X completed!
Press Enter to continue to next level, or Ctrl+C to stop...
```

## Troubleshooting

### Common Issues

1. **"keyboard module not available"**
   ```bash
   pip install keyboard
   ```

2. **"torch not found"**
   - Ensure virtual environment is activated
   - Install PyTorch if missing

3. **Missing model files**
   - Ensure `PPO_preTrained/UAVEnv/` contains the `.pth` files
   - Train models if they don't exist

4. **MuJoCo viewer issues**
   - Install proper graphics drivers
   - Ensure display is available

### Performance Notes

- Manual control requires keyboard focus on the terminal/console window
- Agent trials run with MuJoCo viewer for visual feedback
- Each trial has a 5000-step timeout to prevent infinite loops
- Results are immediately saved after each level

## Expected Workflow

1. **System starts** and checks available models
2. **Level 1 begins** with environment setup (1 obstacle)
3. **Human Expert trial**: Manual control window opens
   - User navigates UAV to goal using keyboard
   - Performance metrics are automatically tracked
   - Trial ends on goal reach, collision, or timeout
4. **Neural Only trial**: PPO agent runs automatically
   - MuJoCo viewer shows agent behavior
   - No user input required
5. **Neurosymbolic trial**: PPO+RDR agent runs automatically
   - Similar to neural but with symbolic rule integration
6. **Level summary** shows comparative results
7. **User prompt** to continue to next level
8. **Repeat** for levels 2-10
9. **Final summary** and CSV export

This system provides comprehensive comparison data to evaluate human expertise versus AI approaches across increasing complexity levels.
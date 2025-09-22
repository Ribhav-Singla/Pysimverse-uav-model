# UAV Navigation with Reinforcement Learning

This project uses a Proximal Policy Optimization (PPO) agent to train a UAV to navigate from a starting position to a goal in a simulated environment with obstacles. The simulation is built using MuJoCo and the agent is trained using PyTorch.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ribhav-Singla/Pysimverse-uav-model.git
    cd Pysimverse-uav-model
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv pysim_env
    source pysim_env/bin/activate  # On Windows, use `pysim_env\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is not present, but this is a standard step. The main dependencies are `mujoco`, `gymnasium`, `torch`, and `numpy`)*

## Usage

### Training the Agent

To train the PPO agent, run the `training.py` script:

```bash
python training.py
```

This will start the training process. The script will periodically save the trained model to the `PPO_preTrained/UAVEnv/` directory. At the end of the training, it will also generate a plot `training_plots.png` showing the rewards and episode lengths over time.

### Visualizing the Agent

To watch the trained agent in action, run the `uav_render.py` script:

```bash
python uav_render.py
```

This script will load the most recently trained model and run the simulation with the agent controlling the UAV.

## File Descriptions

-   `uav_env.py`: Defines the custom `gymnasium` environment for the UAV, including the observation and action spaces, reward function, and simulation logic.
-   `ppo_agent.py`: Implements the PPO agent, including the actor-critic neural network architecture.
-   `training.py`: The main script for training the PPO agent. It handles the training loop, logging, and saving the model.
-   `uav_render.py`: A script for visualizing the performance of a trained agent in the MuJoCo viewer.
-   `environment.xml`: The MuJoCo model file that defines the simulation world, including the UAV and obstacles.

## Configuration

The main simulation and training parameters can be adjusted in the `CONFIG` dictionary at the top of `uav_render.py` and `uav_env.py`. These include:

-   Start and goal positions
-   Number and size of obstacles
-   Reward function parameters
-   Hyperparameters for the PPO agent (in `training.py`)

## Dependencies

-   `mujoco`
-   `gymnasium`
-   `torch`
-   `numpy`
-   `matplotlib`

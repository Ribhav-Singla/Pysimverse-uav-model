import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import math
import random
import csv
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

# Configuration from uav_render.py, adapted for the environment
CONFIG = {
    'start_pos': np.array([-3.0, -3.0, 1.0]),  # Default start position (will be updated dynamically)
    'goal_pos': np.array([3.0, 3.0, 1.0]),     # Default goal position (will be updated dynamically)
    'world_size': 8.0,
    'obstacle_height': 2.0,
    'uav_flight_height': 1.0,  # Half of obstacle height
    'static_obstacles': 9,
    'min_obstacle_size': 0.05,
    'max_obstacle_size': 0.12,
    'collision_distance': 0.1,
    'control_dt': 0.05,
    'max_steps': 10000,  # Max steps per episode
    'boundary_penalty': -100,  # Penalty for going out of bounds
    'lidar_range': 2.9,  # LIDAR maximum detection range
    'lidar_num_rays': 16,  # Number of LIDAR rays (360 degrees)
    'step_reward': -0.01,    # Survival bonus per timestep
}

class RDRRule:
    """A single rule node in the RDR (Ripple Down Rules) system"""
    
    def __init__(self, rule_id: str, condition: Callable, conclusion: str, 
                 action_params: Dict[str, float], parent: Optional['RDRRule'] = None):
        self.rule_id = rule_id
        self.condition = condition  # Function that evaluates to True/False
        self.conclusion = conclusion  # Human-readable rule conclusion
        self.action_params = action_params  # Parameters for action generation
        self.parent = parent
        self.exceptions: List['RDRRule'] = []  # Child rules (exceptions)
        self.usage_count = 0
        self.success_count = 0
        
    def add_exception(self, exception_rule: 'RDRRule'):
        """Add an exception rule to this rule"""
        exception_rule.parent = self
        self.exceptions.append(exception_rule)
        
    def evaluate_condition(self, context: Dict[str, Any]) -> bool:
        """Evaluate if this rule's condition is satisfied"""
        try:
            return self.condition(context)
        except Exception as e:
            print(f"Error evaluating rule {self.rule_id}: {e}")
            return False
    
    def get_success_rate(self) -> float:
        """Get the success rate of this rule"""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count
    
    def update_performance(self, success: bool):
        """Update rule performance statistics"""
        self.usage_count += 1
        if success:
            self.success_count += 1

class RDRRuleSystem:
    """Ripple Down Rules system for UAV navigation"""
    
    def __init__(self):
        self.default_rule = None
        self.all_rules: Dict[str, RDRRule] = {}
        self.rule_counter = 0
        self.initialize_default_rules()
        
    def initialize_default_rules(self):
        """Initialize the default rule hierarchy for UAV navigation"""
        
        # Root rule: Default behavior
        self.default_rule = RDRRule(
            rule_id="R0_DEFAULT",
            condition=lambda ctx: True,  # Always true (fallback)
            conclusion="Default: Move slowly toward goal",
            action_params={"speed_multiplier": 0.1, "direction": "goal"}
        )
        self.all_rules["R0_DEFAULT"] = self.default_rule
        
        # Rule 1: Clear path to goal
        rule_clear_path = RDRRule(
            rule_id="R1_CLEAR_PATH",
            condition=self._condition_clear_path,
            conclusion="Clear path: Move fast toward goal",
            action_params={"speed_multiplier": 0.9, "direction": "goal"}
        )
        self.default_rule.add_exception(rule_clear_path)
        self.all_rules["R1_CLEAR_PATH"] = rule_clear_path
        
        # Exception 1.1: Clear path but near obstacle
        rule_clear_but_near = RDRRule(
            rule_id="R1_1_CLEAR_BUT_NEAR",
            condition=self._condition_clear_but_near_obstacle,
            conclusion="Clear path but near obstacle: Move moderately toward goal",
            action_params={"speed_multiplier": 0.5, "direction": "goal"}
        )
        rule_clear_path.add_exception(rule_clear_but_near)
        self.all_rules["R1_1_CLEAR_BUT_NEAR"] = rule_clear_but_near
        
        # Exception 1.2: Clear path but near boundary
        rule_clear_but_boundary = RDRRule(
            rule_id="R1_2_CLEAR_BUT_BOUNDARY",
            condition=self._condition_clear_but_near_boundary,
            conclusion="Clear path but near boundary: Reduce speed and adjust direction",
            action_params={"speed_multiplier": 0.3, "direction": "goal_adjusted"}
        )
        rule_clear_path.add_exception(rule_clear_but_boundary)
        self.all_rules["R1_2_CLEAR_BUT_BOUNDARY"] = rule_clear_but_boundary
        
        # Rule 2: Blocked path - explore around obstacle
        rule_blocked_path = RDRRule(
            rule_id="R2_BLOCKED_PATH",
            condition=self._condition_blocked_path,
            conclusion="Blocked path: Explore around obstacle",
            action_params={"speed_multiplier": 0.2, "direction": "explore"}
        )
        self.default_rule.add_exception(rule_blocked_path)
        self.all_rules["R2_BLOCKED_PATH"] = rule_blocked_path
        
        # Exception 2.1: Blocked and in corner
        rule_blocked_corner = RDRRule(
            rule_id="R2_1_BLOCKED_CORNER",
            condition=self._condition_blocked_and_cornered,
            conclusion="Blocked and cornered: Backup and then explore",
            action_params={"speed_multiplier": 0.15, "direction": "backup_explore"}
        )
        rule_blocked_path.add_exception(rule_blocked_corner)
        self.all_rules["R2_1_BLOCKED_CORNER"] = rule_blocked_corner
        
        # Exception 2.2: Blocked but goal very close
        rule_blocked_goal_close = RDRRule(
            rule_id="R2_2_BLOCKED_GOAL_CLOSE",
            condition=self._condition_blocked_but_goal_close,
            conclusion="Blocked but goal close: Careful navigation",
            action_params={"speed_multiplier": 0.1, "direction": "careful_goal"}
        )
        rule_blocked_path.add_exception(rule_blocked_goal_close)
        self.all_rules["R2_2_BLOCKED_GOAL_CLOSE"] = rule_blocked_goal_close
        
        # Rule 3: Emergency situations
        rule_emergency = RDRRule(
            rule_id="R3_EMERGENCY",
            condition=self._condition_emergency,
            conclusion="Emergency: Immediate avoidance",
            action_params={"speed_multiplier": 0.05, "direction": "avoid"}
        )
        self.default_rule.add_exception(rule_emergency)
        self.all_rules["R3_EMERGENCY"] = rule_emergency
        
        print(f"🔧 RDR System Initialized with {len(self.all_rules)} rules")
    
    # =====================
    # Condition Functions
    # =====================
    
    def _condition_clear_path(self, ctx: Dict[str, Any]) -> bool:
        """Check if there's a CONFIDENT clear path to goal - high confidence required"""
        return (ctx.get('has_los_to_goal', False) and 
                ctx.get('min_obstacle_dist', 0) > 1.2 and      # Far from obstacles (high confidence)
                ctx.get('distance_to_boundary', 0) > 1.5 and   # Far from boundaries  
                ctx.get('goal_distance', float('inf')) > 1.0)  # Not too close to goal (avoid jitter)
    
    def _condition_clear_but_near_obstacle(self, ctx: Dict[str, Any]) -> bool:
        """Clear path but dangerously close to obstacles - specific danger zone"""
        return (ctx.get('has_los_to_goal', False) and 
                0.5 < ctx.get('min_obstacle_dist', float('inf')) < 1.2 and  # Specific danger range
                ctx.get('distance_to_boundary', 0) > 1.0)  # Not near boundary (different rule)
    
    def _condition_clear_but_near_boundary(self, ctx: Dict[str, Any]) -> bool:
        """Clear path but close to world boundary - boundary-specific issue"""
        return (ctx.get('has_los_to_goal', False) and 
                ctx.get('distance_to_boundary', float('inf')) < 1.0 and
                ctx.get('min_obstacle_dist', 0) > 0.8)  # Not also near obstacles
    
    def _condition_blocked_path(self, ctx: Dict[str, Any]) -> bool:
        """Path to goal is blocked - confident blocking with safe distance"""
        return (not ctx.get('has_los_to_goal', False) and
                ctx.get('min_obstacle_dist', 0) > 0.8 and      # Not in emergency zone
                ctx.get('distance_to_boundary', 0) > 1.0 and   # Not near boundaries
                ctx.get('num_blocked_directions', 0) < 3)      # Not cornered (different rule)
    
    def _condition_blocked_and_cornered(self, ctx: Dict[str, Any]) -> bool:
        """Blocked path and UAV is in a corner/tight space - specific cornering situation"""
        return (not ctx.get('has_los_to_goal', False) and
                (ctx.get('num_blocked_directions', 0) >= 3 or   # Many blocked directions OR
                 ctx.get('distance_to_boundary', float('inf')) < 0.8))  # Very close to boundary
    
    def _condition_blocked_but_goal_close(self, ctx: Dict[str, Any]) -> bool:
        """Path blocked but goal is very close - precision navigation required"""
        return (not ctx.get('has_los_to_goal', False) and
                ctx.get('goal_distance', float('inf')) < 1.2 and  # Goal is close
                ctx.get('min_obstacle_dist', 0) > 0.5)  # Not in emergency
    
    def _condition_emergency(self, ctx: Dict[str, Any]) -> bool:
        """Emergency situation - immediate collision danger"""
        return ctx.get('min_obstacle_dist', float('inf')) < 0.4  # Increased threshold for emergency
    
    def evaluate_rules(self, context: Dict[str, Any]) -> RDRRule:
        """Evaluate the rule hierarchy and return the most specific applicable rule"""
        return self._evaluate_rule_recursive(self.default_rule, context)
    
    def has_specific_rule(self, context: Dict[str, Any]) -> bool:
        """Check if any specific (non-default) rule applies to the context"""
        # Use the actual evaluation process and check if result is not default
        applicable_rule = self.evaluate_rules(context)
        return applicable_rule.rule_id != "R0_DEFAULT"
    
    def _evaluate_rule_recursive(self, rule: RDRRule, context: Dict[str, Any]) -> RDRRule:
        """Recursively evaluate rules, checking exceptions first"""
        
        # Check all exceptions (more specific rules)
        for exception in rule.exceptions:
            if exception.evaluate_condition(context):
                # Recursively check if this exception has more specific exceptions
                return self._evaluate_rule_recursive(exception, context)
        
        # If no exceptions apply, return this rule (if its condition is met)
        if rule.evaluate_condition(context):
            return rule
        
        # This shouldn't happen if default rule is properly set up
        return self.default_rule
    
    def add_new_rule(self, parent_rule_id: str, condition: Callable, 
                     conclusion: str, action_params: Dict[str, float]) -> str:
        """Add a new exception rule to an existing rule"""
        self.rule_counter += 1
        new_rule_id = f"R{self.rule_counter}_LEARNED"
        
        parent_rule = self.all_rules.get(parent_rule_id)
        if parent_rule is None:
            print(f"Warning: Parent rule {parent_rule_id} not found")
            return None
        
        new_rule = RDRRule(
            rule_id=new_rule_id,
            condition=condition,
            conclusion=conclusion,
            action_params=action_params,
            parent=parent_rule
        )
        
        parent_rule.add_exception(new_rule)
        self.all_rules[new_rule_id] = new_rule
        
        print(f"📚 Added new RDR rule: {new_rule_id} -> {conclusion}")
        return new_rule_id
    
    def get_rule_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all rules"""
        stats = {}
        for rule_id, rule in self.all_rules.items():
            stats[rule_id] = {
                'usage_count': rule.usage_count,
                'success_count': rule.success_count,
                'success_rate': rule.get_success_rate()
            }
        return stats
    
    def print_rule_hierarchy(self, rule: RDRRule = None, indent: int = 0):
        """Print the rule hierarchy for debugging"""
        if rule is None:
            rule = self.default_rule
            print("🌳 RDR Rule Hierarchy:")
        
        prefix = "  " * indent + ("├─ " if indent > 0 else "")
        success_rate = f"({rule.get_success_rate():.2f})" if rule.usage_count > 0 else "(unused)"
        print(f"{prefix}{rule.rule_id}: {rule.conclusion} {success_rate}")
        
        for exception in rule.exceptions:
            self.print_rule_hierarchy(exception, indent + 1)

class EnvironmentGenerator:
    @staticmethod
    def generate_obstacles(num_obstacles=None):
        """Generate obstacles with specified count for curriculum learning"""
        if num_obstacles is None:
            num_obstacles = CONFIG['static_obstacles']
        
        obstacles = []
        world_size = CONFIG['world_size']
        half_world = world_size / 2
        
        # Generate potential positions with grid-based distribution
        grid_size = int(math.sqrt(max(num_obstacles, 9))) + 1
        cell_size = world_size / grid_size
        positions = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                x = -half_world + (i + 0.5) * cell_size
                y = -half_world + (j + 0.5) * cell_size
                positions.append((x, y))
        
        random.shuffle(positions)
        
        for i in range(num_obstacles):
            x, y = positions[i % len(positions)]
            
            # Add some randomness to avoid perfect grid alignment
            x += random.uniform(-cell_size/4, cell_size/4)
            y += random.uniform(-cell_size/4, cell_size/4)
            
            # Ensure obstacles stay within bounds
            x = max(-half_world + 0.5, min(half_world - 0.5, x))
            y = max(-half_world + 0.5, min(half_world - 0.5, y))
            
            size_x = random.uniform(CONFIG['min_obstacle_size'], CONFIG['max_obstacle_size'])
            size_y = random.uniform(CONFIG['min_obstacle_size'], CONFIG['max_obstacle_size'])
            height = CONFIG['obstacle_height']
            
            obstacle_type = random.choice(['box', 'cylinder'])
            color = [random.uniform(0.1, 0.9) for _ in range(3)] + [1.0]
            
            obstacles.append({
                'type': 'static',
                'shape': obstacle_type,
                'pos': [x, y, height/2],
                'size': [size_x, size_y, height/2] if obstacle_type == 'box' else [min(size_x, size_y), height/2],
                'color': color,
                'id': f'static_obs_{i}'
            })
        
        return obstacles

    @staticmethod
    def get_random_corner_position():
        """Get a random start position from one of the four corners"""
        half_world = CONFIG['world_size'] / 2 - 1.0  # 1m margin from boundary
        corners = [
            np.array([-half_world, -half_world, CONFIG['uav_flight_height']]),  # Bottom-left
            np.array([half_world, -half_world, CONFIG['uav_flight_height']]),   # Bottom-right
            np.array([-half_world, half_world, CONFIG['uav_flight_height']]),   # Top-left
            np.array([half_world, half_world, CONFIG['uav_flight_height']])     # Top-right
        ]
        return random.choice(corners)
    
    @staticmethod
    def get_random_goal_position():
        """Get a random goal position anywhere in the map (not just corners)"""
        half_world = CONFIG['world_size'] / 2 - 1.0  # 1m margin from boundary
        x = random.uniform(-half_world, half_world)
        y = random.uniform(-half_world, half_world)
        return np.array([x, y, CONFIG['uav_flight_height']])
    
    @staticmethod
    def check_position_safety(position, obstacles, safety_radius=0.8):
        """Check if a position is safe from obstacles"""
        for obs in obstacles:
            obs_pos = np.array(obs['pos'])
            distance = np.linalg.norm(position[:2] - obs_pos[:2])  # Only check horizontal distance
            
            if obs['shape'] == 'box':
                # For box obstacles, consider the size
                min_safe_distance = max(obs['size'][0], obs['size'][1]) + safety_radius
            elif obs['shape'] == 'cylinder':
                # For cylinder obstacles
                min_safe_distance = obs['size'][0] + safety_radius
            else:  # sphere
                min_safe_distance = obs['size'][0] + safety_radius
            
            if distance < min_safe_distance:
                return False
        return True

    @staticmethod
    def generate_curriculum_maps():
        """Generate 50 maps each for obstacle counts 1-10 with versatile start/goal positions"""
        curriculum_maps = {}
        
        for obstacle_count in range(1, 11):  # 1 to 10 obstacles
            maps = []
            for map_id in range(50):  # 50 maps per obstacle count (increased from 20)
                # Set random seed for reproducible map generation
                random.seed(obstacle_count * 1000 + map_id)
                
                # Generate obstacles first
                obstacles = EnvironmentGenerator.generate_obstacles(obstacle_count)
                
                # Generate safe start and goal positions
                max_attempts = 50
                safety_radius = 0.8
                
                # Get random start position from corners
                start_pos = None
                for attempt in range(max_attempts):
                    candidate_start = EnvironmentGenerator.get_random_corner_position()
                    if EnvironmentGenerator.check_position_safety(candidate_start, obstacles, safety_radius):
                        start_pos = candidate_start
                        break
                
                # If no safe corner found, use default corner and warn
                if start_pos is None:
                    start_pos = np.array([-3.0, -3.0, 1.0])
                    print(f"⚠️ Warning: Using default start position for map {obstacle_count}-{map_id}")
                
                # Get random goal position anywhere in map
                goal_pos = None
                for attempt in range(max_attempts):
                    candidate_goal = EnvironmentGenerator.get_random_goal_position()
                    # Ensure goal is not too close to start position
                    start_goal_distance = np.linalg.norm(candidate_goal[:2] - start_pos[:2])
                    if (EnvironmentGenerator.check_position_safety(candidate_goal, obstacles, safety_radius) and 
                        start_goal_distance > 2.0):  # At least 2m apart
                        goal_pos = candidate_goal
                        break
                
                # If no safe goal found, use opposite corner from start
                if goal_pos is None:
                    goal_pos = np.array([3.0, 3.0, 1.0])
                    print(f"⚠️ Warning: Using default goal position for map {obstacle_count}-{map_id}")
                
                maps.append({
                    'obstacles': obstacles,
                    'start_pos': start_pos,
                    'goal_pos': goal_pos,
                    'obstacle_count': obstacle_count,
                    'map_id': map_id
                })
            curriculum_maps[obstacle_count] = maps
        
        # Reset random seed
        random.seed()
        return curriculum_maps

    @staticmethod
    def create_xml_with_obstacles(obstacles):
        xml_template = f'''<mujoco model="uav_env">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="1 1" reflectance="0.2"/>
  </asset>
  <worldbody>
    <geom name="ground" type="plane" size="{CONFIG['world_size']/2} {CONFIG['world_size']/2} 0.1" material="grid"/>
    <light name="light1" pos="0 0 4" dir="0 0 -1" diffuse="1 1 1"/>
    <body name="chassis" pos="{CONFIG['start_pos'][0]} {CONFIG['start_pos'][1]} {CONFIG['start_pos'][2]}">
      <joint type="free" name="root"/>
      <geom type="box" size="0.12 0.12 0.02" rgba="1.0 0.0 0.0 1.0" mass="0.8"/>
      <site name="motor1" pos="0.08 0.08 0" size="0.01"/>
      <site name="motor2" pos="-0.08 0.08 0" size="0.01"/>
      <site name="motor3" pos="0.08 -0.08 0" size="0.01"/>
      <site name="motor4" pos="-0.08 -0.08 0" size="0.01"/>
    </body>'''
        for obs in obstacles:
            if obs['shape'] == 'box':
                xml_template += f'''
    <geom name="{obs['id']}" type="box" size="{obs['size'][0]} {obs['size'][1]} {obs['size'][2]}" pos="{obs['pos'][0]} {obs['pos'][1]} {obs['pos'][2]}" rgba="{obs['color'][0]} {obs['color'][1]} {obs['color'][2]} {obs['color'][3]}"/>'''
            elif obs['shape'] == 'cylinder':
                xml_template += f'''
    <geom name="{obs['id']}" type="cylinder" size="{obs['size'][0]} {obs['size'][1]}" pos="{obs['pos'][0]} {obs['pos'][1]} {obs['pos'][2]}" rgba="{obs['color'][0]} {obs['color'][1]} {obs['color'][2]} {obs['color'][3]}"/>'''
        xml_template += '''
  </worldbody>
  <actuator>
    <motor name="m1" gear="0 0 1 0 0 0" site="motor1"/>
    <motor name="m2" gear="0 0 1 0 0 0" site="motor2"/>
    <motor name="m3" gear="0 0 1 0 0 0" site="motor3"/>
    <motor name="m4" gear="0 0 1 0 0 0" site="motor4"/>
  </actuator>
</mujoco>'''
        with open("environment.xml", 'w') as f:
            f.write(xml_template)
        return obstacles

class UAVEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None, curriculum_learning=False, ns_cfg=None):
        super().__init__()
        self.render_mode = render_mode
        self.curriculum_learning = curriculum_learning
        # Neurosymbolic configuration (optional, with safe defaults)
        self.ns_cfg = ns_cfg if ns_cfg is not None else {
            'use_neurosymbolic': False,
            'lambda': 0.0,
            'warmup_steps': 20,
            'high_speed': 0.9,
            'blocked_strength': 0.1
        }
        
        # Initialize curriculum learning system
        if self.curriculum_learning:
            self.curriculum_maps = EnvironmentGenerator.generate_curriculum_maps()
            self.current_obstacle_level = 1
            self.current_map_pool = self.curriculum_maps[1]
            print(f"🎓 Curriculum Learning Initialized: Generated maps for obstacle levels 1-10")
            print(f"   - 50 maps per obstacle level (500 total maps)")
            print(f"   - Random start positions from corners")
            print(f"   - Random goal positions anywhere in map")
            print(f"   - Safe positioning with 0.8m obstacle clearance")
            print(f"   - Starting with obstacle level: {self.current_obstacle_level}")
        
        # Generate initial obstacles
        self.obstacles = EnvironmentGenerator.generate_obstacles() if not curriculum_learning else []
        self.load_curriculum_map() if curriculum_learning else EnvironmentGenerator.create_xml_with_obstacles(self.obstacles)
        
        self.model = mujoco.MjModel.from_xml_path("environment.xml")
        self.data = mujoco.MjData(self.model)
        
        # Observation space: [pos(3), vel(3), goal_dist(3), lidar_readings(16), lidar_features(11)]
        # LIDAR features: min, mean, closest_dir(2), danger_level, clearances(4), goal_alignment(2)
        obs_dim = 3 + 3 + 3 + CONFIG['lidar_num_rays'] + 11
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Action space: 3D velocity control (vx, vy, vz=0) - no Z-axis movement
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(3,), dtype=np.float32)
        
        self.viewer = None
        self.step_count = 0
        
        # Initialize CSV logging for obstacle detection
        self.csv_log_path = "obstacle_detection_log.csv"
        self.init_csv_logging()
        
        # Store previous velocity for trajectory change calculation
        self.prev_velocity = np.zeros(3)
        
        # Episode counter for adaptive thresholds
        self.current_episode = 0
        
        # Goal stabilization removed - simple goal achievement

        # Episode-local timestep counter (for neurosymbolic warmup logic)
        self._episode_timestep = 0
        # Neurosymbolic LOS confirmation and cooldown tracking
        self._ns_los_confirm_count = 0
        self._ns_cooldown_steps_remaining = 0
        
        # Initialize RDR (Ripple Down Rules) system
        self.rdr_system = RDRRuleSystem()
        self.current_rule = None  # Track which rule was used last
        self.rule_performance_tracking = {}  # Track rule success/failure

    
    
    def step(self, action):
        # Track per-episode timestep
        self._episode_timestep += 1
        # Store previous velocity for trajectory change calculation
        self.prev_velocity = self.data.qvel[:3].copy()
        
        # Action is now 3D: [vx, vy, vz] but we ignore vz and keep constant height
        # Convert action to numpy array if it's a tensor
        if hasattr(action, 'numpy'):
            action = action.numpy()
        action = np.array(action).flatten()
        
        # Add goal-directed bias to action in early training
        # Restore baseline behavior when neurosymbolic is off; use non-negative only when active
        if self.current_episode < 200:  # Only in early training
            pos = self.data.qpos[:3]
            goal_vector = CONFIG['goal_pos'] - pos
            goal_direction = goal_vector / (np.linalg.norm(goal_vector) + 1e-8)  # Normalized goal direction
            bias_strength = max(0, (200 - self.current_episode) / 200)  # Fade out bias over first 200 episodes
            # Allow full signed directions [-1, 1] in both neurosymbolic and baseline modes
            action[:2] = (1 - bias_strength * 0.5) * action[:2] + (bias_strength * 0.5) * goal_direction[:2]
        
        # Direct velocity control - no constraints, full [-1, 1] range
        target_vel = action[:2]  # Desired velocity from action
        
        # Simple velocity control - direct action mapping
        self.data.qvel[0] = float(target_vel[0])  # X velocity  
        self.data.qvel[1] = float(target_vel[1])  # Y velocity
        self.data.qvel[2] = 0.0                   # Z velocity = 0 (no vertical movement)
        
        # Maintain constant height
        self.data.qpos[2] = CONFIG['uav_flight_height']
        
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        # No velocity constraints - allow full [-1, 1] range
        
        # Always ensure Z position stays constant
        self.data.qpos[2] = CONFIG['uav_flight_height']
        self.data.qvel[2] = 0.0
        
        obs = self._get_obs()
        # Update near-miss cooldown when neurosymbolic is active
        if self.ns_cfg.get('use_neurosymbolic', False) and float(self.ns_cfg.get('lambda', 0.0)) >= 1.0:
            lidar_vals = obs[9:25]
            min_norm = float(np.min(lidar_vals))
            lidar_min_m = min_norm * CONFIG['lidar_range']
            near_miss_thresh = float(self.ns_cfg.get('near_miss_threshold_m', 0.5))
            cooldown_steps = int(self.ns_cfg.get('near_miss_cooldown_steps', 10))
            if lidar_min_m < near_miss_thresh:
                self._ns_cooldown_steps_remaining = cooldown_steps
            elif self._ns_cooldown_steps_remaining > 0:
                self._ns_cooldown_steps_remaining -= 1
        reward, termination_info = self._get_reward_and_termination_info(obs)
        terminated = termination_info['terminated']
        truncated = self.step_count >= CONFIG['max_steps']
        
        # Update RDR rule performance based on step outcome
        if hasattr(self, 'current_rule') and self.current_rule is not None:
            # Define success criteria for RDR rule evaluation
            rule_success = (
                reward > 0 or  # Positive reward
                (not termination_info['collision_detected'] and 
                 not termination_info['out_of_bounds'] and
                 self.prev_goal_dist - np.linalg.norm(CONFIG['goal_pos'] - obs[:3]) > 0)  # Made progress
            )
            self.update_rule_performance(rule_success)
        
        # Store termination info for logging
        self.last_termination_info = termination_info
        if truncated and not terminated:
            self.last_termination_info['termination_reason'] = 'max_steps_reached'
            self.last_termination_info['terminated'] = True
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._episode_timestep = 0
        self._ns_los_confirm_count = 0
        self._ns_cooldown_steps_remaining = 0
        
        # Set dynamic goal position (randomly select from three available corners)
        CONFIG['goal_pos'] = self._get_random_goal_position()
        
        # Reset UAV position and state with constant height
        self.data.qpos[:3] = CONFIG['start_pos']
        self.data.qpos[2] = CONFIG['uav_flight_height']  # Ensure constant height
        self.data.qvel[:] = 0
        
        # Reset velocity tracking
        self.prev_velocity = np.zeros(3)
        
        # Initialize termination info
        self.last_termination_info = {
            'terminated': False,
            'termination_reason': 'none',
            'final_position': None,
            'goal_distance': None,
            'collision_detected': False,
            'out_of_bounds': False
        }
        
        # Goal stabilization removed - no tracking needed
        
        # Reset RDR tracking
        self.current_rule = None
        
        # Regenerate obstacles for each episode
        if self.curriculum_learning:
            self.load_curriculum_map()
        else:
            self.obstacles = EnvironmentGenerator.generate_obstacles()
            EnvironmentGenerator.create_xml_with_obstacles(self.obstacles)
        
        self.model = mujoco.MjModel.from_xml_path("environment.xml")
        self.data = mujoco.MjData(self.model)
        
        # Set initial position again after model reload
        self.data.qpos[:3] = CONFIG['start_pos']
        
        # INITIALIZE WITH SMALL POSITIVE X VELOCITY to avoid early westward movement
        if self.current_episode < 300:  # Only for early training
            # Small initial eastward velocity (will help avoid immediate west boundary issues)
            self.data.qvel[0] = 0.05  # Small positive X velocity (eastward)
            self.data.qvel[1] = 0.0   # No Y velocity
            self.data.qvel[2] = 0.0   # No Z velocity
        self.data.qpos[2] = CONFIG['uav_flight_height']
        
        # Initialize previous goal distance
        self.prev_goal_dist = np.linalg.norm(CONFIG['goal_pos'] - self.data.qpos[:3])
        
        return self._get_obs(), {}

    # =====================
    # Neurosymbolic helpers
    # =====================
    def get_goal_vector(self):
        """Return unit vector from UAV to goal (2D XY) and distance."""
        pos = self.data.qpos[:3]
        vec = CONFIG['goal_pos'] - pos
        dist = float(np.linalg.norm(vec[:2]) + 1e-8)
        dir_xy = vec[:2] / dist
        return dir_xy, dist

    def has_line_of_sight_to_goal(self):
        """LOS check with angular safety margin and confirmation steps; disabled during cooldown."""
        pos = self.data.qpos[:3]
        goal_vec = CONFIG['goal_pos'] - pos
        goal_dist = float(np.linalg.norm(goal_vec[:2]))
        if goal_dist <= 1e-6:
            return True
        # Disable LOS during cooldown
        if self._ns_cooldown_steps_remaining > 0:
            self._ns_los_confirm_count = 0
            return False
        # Angular margin (degrees -> radians)
        ang_margin_deg = float(self.ns_cfg.get('los_angle_margin_deg', 5.0))
        ang_margin = np.radians(ang_margin_deg)
        # Base ray
        ray_dir = np.array([goal_vec[0], goal_vec[1], 0.0]) / (goal_dist + 1e-8)
        # Rotate helper
        def rot(vec, ang):
            c, s = np.cos(ang), np.sin(ang)
            return np.array([c*vec[0]-s*vec[1], s*vec[0]+c*vec[1], 0.0])
        rays = [ray_dir, rot(ray_dir, ang_margin), rot(ray_dir, -ang_margin)]
        min_obs_dist = CONFIG['lidar_range']
        for rd in rays:
            for obs in self.obstacles:
                obs_dist = self._ray_obstacle_intersection(pos, rd, obs)
                if obs_dist < min_obs_dist:
                    min_obs_dist = obs_dist
        los_now = (min_obs_dist >= goal_dist - 1e-6)
        confirm_k = int(self.ns_cfg.get('los_confirm_steps', 3))
        if los_now:
            self._ns_los_confirm_count = min(self._ns_los_confirm_count + 1, confirm_k)
        else:
            self._ns_los_confirm_count = 0
        return self._ns_los_confirm_count >= confirm_k

    def symbolic_action(self, t_step=None):
        """Compute RDR-based action in env action space (vx, vy, vz=0)."""
        if t_step is None:
            t_step = self._episode_timestep
        
        # Prepare context for RDR system
        pos = self.data.qpos[:3]
        obs = self._get_obs()
        lidar_readings = obs[9:25]  # LIDAR data
        
        context = self._prepare_rdr_context(pos, lidar_readings, obs)
        
        # Evaluate RDR system to get applicable rule
        applicable_rule = self.rdr_system.evaluate_rules(context)
        self.current_rule = applicable_rule
        
        # Generate action based on rule
        action = self._generate_action_from_rule(applicable_rule, context)
        
        # Update rule usage
        applicable_rule.usage_count += 1
        
        if self.ns_cfg.get('debug_rdr', False):
            print(f"🔍 RDR: Applied rule {applicable_rule.rule_id} - {applicable_rule.conclusion}")
        
        return action
    
    def _prepare_rdr_context(self, pos: np.ndarray, lidar_readings: np.ndarray, obs: np.ndarray) -> Dict[str, Any]:
        """Prepare context dictionary for RDR rule evaluation"""
        
        # Basic positioning
        goal_vector = CONFIG['goal_pos'] - pos
        goal_distance = float(np.linalg.norm(goal_vector[:2]))
        
        # LIDAR analysis
        min_obstacle_dist = float(np.min(lidar_readings) * CONFIG['lidar_range'])
        mean_obstacle_dist = float(np.mean(lidar_readings) * CONFIG['lidar_range'])
        
        # Count blocked directions (LIDAR readings below threshold)
        blocked_threshold = 0.3  # 30% of max LIDAR range
        num_blocked_directions = int(np.sum(lidar_readings < blocked_threshold))
        
        # Boundary analysis
        half_world = CONFIG['world_size'] / 2
        distances_to_boundaries = np.array([
            half_world + pos[0],  # Distance to west boundary
            half_world - pos[0],  # Distance to east boundary
            half_world + pos[1],  # Distance to south boundary
            half_world - pos[1]   # Distance to north boundary
        ])
        distance_to_boundary = float(np.min(distances_to_boundaries))
        
        # Directional clearance analysis
        sector_size = len(lidar_readings) // 4
        clearances = {
            'front': float(np.mean(lidar_readings[0:sector_size])),
            'right': float(np.mean(lidar_readings[sector_size:2*sector_size])),
            'back': float(np.mean(lidar_readings[2*sector_size:3*sector_size])),
            'left': float(np.mean(lidar_readings[3*sector_size:4*sector_size]))
        }
        
        return {
            'position': pos,
            'goal_vector': goal_vector,
            'goal_distance': goal_distance,
            'has_los_to_goal': self.has_line_of_sight_to_goal(),
            'min_obstacle_dist': min_obstacle_dist,
            'mean_obstacle_dist': mean_obstacle_dist,
            'num_blocked_directions': num_blocked_directions,
            'distance_to_boundary': distance_to_boundary,
            'lidar_readings': lidar_readings,
            'directional_clearances': clearances,
            'velocity': self.data.qvel[:3],
            'episode_step': self._episode_timestep
        }
    
    def _generate_action_from_rule(self, rule: RDRRule, context: Dict[str, Any]) -> np.ndarray:
        """Generate action based on RDR rule parameters"""
        
        # Get rule parameters
        speed_multiplier = rule.action_params.get('speed_multiplier', 0.1)
        direction_type = rule.action_params.get('direction', 'goal')
        
        # No velocity constraints - use full action space range
        action_cap = float(np.max(self.action_space.high))  # Should be 1.0
        
        # Distance-aware speed scaling
        goal_distance = context['goal_distance']
        far_dist = 3.0
        min_scale = 0.4
        dist_scale = max(min_scale, min(1.0, goal_distance / max(far_dist, 1e-6)))
        
        # Calculate base speed using full action range
        base_speed = dist_scale * speed_multiplier * action_cap
        
        # Generate direction vector based on rule direction type
        if direction_type == 'goal':
            # Direct toward goal
            dir_xy = context['goal_vector'][:2]
            dir_xy = dir_xy / (np.linalg.norm(dir_xy) + 1e-8)
            
        elif direction_type == 'goal_adjusted':
            # Toward goal but adjusted for boundary avoidance
            dir_xy = context['goal_vector'][:2]
            dir_xy = dir_xy / (np.linalg.norm(dir_xy) + 1e-8)
            
            # Adjust direction away from boundaries
            pos = context['position']
            half_world = CONFIG['world_size'] / 2
            
            # Add repulsion from close boundaries
            if abs(pos[0] + half_world) < 1.0:  # Near west boundary
                dir_xy[0] += 0.5  # Push east
            if abs(pos[0] - half_world) < 1.0:  # Near east boundary
                dir_xy[0] -= 0.5  # Push west
            if abs(pos[1] + half_world) < 1.0:  # Near south boundary
                dir_xy[1] += 0.5  # Push north
            if abs(pos[1] - half_world) < 1.0:  # Near north boundary
                dir_xy[1] -= 0.5  # Push south
            
            dir_xy = dir_xy / (np.linalg.norm(dir_xy) + 1e-8)
            
        elif direction_type == 'explore':
            # Explore around obstacles - find clearest direction
            clearances = context['directional_clearances']
            best_direction = max(clearances, key=clearances.get)
            
            direction_angles = {'front': 0, 'right': np.pi/2, 'back': np.pi, 'left': 3*np.pi/2}
            angle = direction_angles[best_direction]
            dir_xy = np.array([np.cos(angle), np.sin(angle)])
            
        elif direction_type == 'backup_explore':
            # Backup first, then explore
            # Move opposite to current goal direction briefly, then explore
            if context['episode_step'] % 20 < 5:  # Backup phase
                dir_xy = -context['goal_vector'][:2]
                dir_xy = dir_xy / (np.linalg.norm(dir_xy) + 1e-8)
            else:  # Explore phase
                clearances = context['directional_clearances']
                best_direction = max(clearances, key=clearances.get)
                direction_angles = {'front': 0, 'right': np.pi/2, 'back': np.pi, 'left': 3*np.pi/2}
                angle = direction_angles[best_direction]
                dir_xy = np.array([np.cos(angle), np.sin(angle)])
                
        elif direction_type == 'careful_goal':
            # Very careful movement toward goal
            dir_xy = context['goal_vector'][:2]
            dir_xy = dir_xy / (np.linalg.norm(dir_xy) + 1e-8)
            base_speed *= 0.5  # Extra speed reduction
            
        elif direction_type == 'avoid':
            # Emergency avoidance - move away from closest obstacle
            lidar_readings = context['lidar_readings']
            closest_idx = np.argmin(lidar_readings)
            closest_angle = (2 * np.pi * closest_idx) / CONFIG['lidar_num_rays']
            # Move in opposite direction
            avoid_angle = closest_angle + np.pi
            dir_xy = np.array([np.cos(avoid_angle), np.sin(avoid_angle)])
            
        else:
            # Default: move toward goal
            dir_xy = context['goal_vector'][:2]
            dir_xy = dir_xy / (np.linalg.norm(dir_xy) + 1e-8)
        
        # Allow full signed action space [-1, 1] - no restrictions on negative components
        
        # Construct action
        vx = float(base_speed * dir_xy[0])
        vy = float(base_speed * dir_xy[1])
        vz = 0.0
        
        action = np.array([vx, vy, vz], dtype=np.float32)
        
        # Clip to action bounds
        low = np.broadcast_to(self.action_space.low, action.shape)
        high = np.broadcast_to(self.action_space.high, action.shape)
        return np.clip(action, low, high)
    
    def update_rule_performance(self, success: bool):
        """Update performance of the last used RDR rule"""
        if self.current_rule is not None:
            self.current_rule.update_performance(success)
    
    def get_rdr_statistics(self):
        """Get RDR system statistics"""
        return self.rdr_system.get_rule_statistics()
    
    def print_rdr_hierarchy(self):
        """Print the RDR rule hierarchy"""
        self.rdr_system.print_rule_hierarchy()
    
    def has_specific_rdr_rule(self):
        """Check if a specific (non-default) RDR rule is available for current state"""
        pos = self.data.qpos[:3]
        obs = self._get_obs()
        lidar_readings = obs[9:25]  # LIDAR data
        context = self._prepare_rdr_context(pos, lidar_readings, obs)
        return self.rdr_system.has_specific_rule(context)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()
    
    def _get_random_goal_position(self):
        """Select a random goal position from the three available corners (excluding start position)"""
        # Define the four corners of the world with LARGER SAFETY MARGIN
        half_world = CONFIG['world_size'] / 2
        safety_margin = 1.0  # INCREASED: Keep 1.0m away from boundary
        
        corners = [
            np.array([half_world - safety_margin, half_world - safety_margin, CONFIG['uav_flight_height']]),    # Top-right [3, 3]
            np.array([half_world - safety_margin, -half_world + safety_margin, CONFIG['uav_flight_height']]),   # Bottom-right [3, -3]
            np.array([-half_world + safety_margin, half_world - safety_margin, CONFIG['uav_flight_height']])    # Top-left [-3, 3]
        ]
        # Start position is bottom-left: [-3, -3, 1.0]
        # So we exclude it and randomly select from the other three corners
        return random.choice(corners)
    
    def init_csv_logging(self):
        """Initialize CSV file for obstacle detection logging"""
        # Create headers for the CSV file
        headers = [
            'timestamp', 'episode', 'step', 'curriculum_level', 'map_id', 'obstacle_count',
            'start_x', 'start_y', 'start_z', 'goal_x', 'goal_y', 'goal_z',
            'uav_x', 'uav_y', 'uav_z', 'obstacle_x', 'obstacle_y', 'obstacle_z', 
            'obstacle_type', 'obstacle_id', 'detection_distance', 'detection_angle', 
            'prev_velocity_x', 'prev_velocity_y', 'new_velocity_x', 'new_velocity_y', 
            'trajectory_change_angle'
        ]
        
        # Always create/overwrite the CSV file to clear previous logs
        with open(self.csv_log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
        # Initialize curriculum episode logging
        self.curriculum_log_path = "curriculum_learning_log.csv"
        curriculum_headers = [
            'timestamp', 'episode', 'curriculum_level', 'episode_in_level', 'map_id',
            'obstacle_count', 'start_x', 'start_y', 'start_z', 'goal_x', 'goal_y', 'goal_z',
            'episode_reward', 'episode_length', 'termination_reason',
            'goal_reached', 'collision_detected', 'out_of_bounds', 'final_position_x',
            'final_position_y', 'final_position_z', 'goal_distance', 'final_velocity'
        ]
        
        # Always create/overwrite the curriculum log file
        with open(self.curriculum_log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(curriculum_headers)
                
    def log_obstacle_detection(self, uav_pos, obstacle_pos, obstacle_type, obstacle_id, 
                             detection_distance, detection_angle, prev_vel, new_vel):
        """Log obstacle detection event to CSV file"""
        # Calculate trajectory change angle
        prev_vel_2d = prev_vel[:2]
        new_vel_2d = new_vel[:2]
        
        trajectory_change_angle = 0.0
        if np.linalg.norm(prev_vel_2d) > 0.01 and np.linalg.norm(new_vel_2d) > 0.01:
            # Normalize vectors
            prev_vel_norm = prev_vel_2d / np.linalg.norm(prev_vel_2d)
            new_vel_norm = new_vel_2d / np.linalg.norm(new_vel_2d)
            
            # Calculate angle between vectors
            dot_product = np.clip(np.dot(prev_vel_norm, new_vel_norm), -1.0, 1.0)
            trajectory_change_angle = np.degrees(np.arccos(dot_product))
        
        # Write to CSV with curriculum information
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        curriculum_info = self.get_curriculum_info() if self.curriculum_learning else None
        
        row = [
            timestamp, getattr(self, 'current_episode', 0), self.step_count,
            curriculum_info['current_level'] if curriculum_info else 0,
            curriculum_info['current_map_info']['map_id'] if curriculum_info and curriculum_info['current_map_info'] else 0,
            curriculum_info['current_map_info']['obstacle_count'] if curriculum_info and curriculum_info['current_map_info'] else 0,
            CONFIG['start_pos'][0], CONFIG['start_pos'][1], CONFIG['start_pos'][2],
            CONFIG['goal_pos'][0], CONFIG['goal_pos'][1], CONFIG['goal_pos'][2],
            uav_pos[0], uav_pos[1], uav_pos[2],
            obstacle_pos[0], obstacle_pos[1], obstacle_pos[2],
            obstacle_type, obstacle_id, detection_distance, detection_angle,
            prev_vel[0], prev_vel[1], new_vel[0], new_vel[1], trajectory_change_angle
        ]
        
        with open(self.csv_log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
    
    def log_curriculum_episode(self, episode_num, episode_in_level, episode_reward, 
                             episode_length, termination_info):
        """Log curriculum learning episode data"""
        if not self.curriculum_learning:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        curriculum_info = self.get_curriculum_info()
        map_info = curriculum_info['current_map_info'] if curriculum_info else {}
        
        # Extract termination information
        final_pos = termination_info.get('final_position', [0, 0, 0])
        goal_reached = termination_info.get('termination_reason') == 'goal_reached'
        collision = termination_info.get('collision_detected', False)
        out_of_bounds = termination_info.get('out_of_bounds', False)
        
        row = [
            timestamp, episode_num, 
            curriculum_info['current_level'] if curriculum_info else 0,
            episode_in_level,
            map_info.get('map_id', 0),
            map_info.get('obstacle_count', 0),
            CONFIG['start_pos'][0], CONFIG['start_pos'][1], CONFIG['start_pos'][2],
            CONFIG['goal_pos'][0], CONFIG['goal_pos'][1], CONFIG['goal_pos'][2],
            episode_reward, episode_length,
            termination_info.get('termination_reason', 'unknown'),
            goal_reached, collision, out_of_bounds,
            final_pos[0] if len(final_pos) > 0 else 0,
            final_pos[1] if len(final_pos) > 1 else 0,
            final_pos[2] if len(final_pos) > 2 else 0,
            termination_info.get('goal_distance', 0),
            termination_info.get('final_velocity', 0)
        ]
        
        with open(self.curriculum_log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
    
    def set_curriculum_level(self, obstacle_level):
        """Set the current curriculum level (1-10 obstacles)"""
        if not self.curriculum_learning:
            print("⚠️  Curriculum learning not enabled for this environment")
            return
            
        if obstacle_level < 1 or obstacle_level > 10:
            print(f"⚠️  Invalid obstacle level: {obstacle_level}. Must be between 1-10")
            return
            
        self.current_obstacle_level = obstacle_level
        self.current_map_pool = self.curriculum_maps[obstacle_level]
        print(f"🎓 Curriculum Level Updated: Now using {obstacle_level} obstacle maps")
    
    def load_curriculum_map(self):
        """Load a random map from the current curriculum level"""
        if not self.curriculum_learning:
            return
            
        # Select random map from current level
        selected_map = random.choice(self.current_map_pool)
        self.obstacles = selected_map['obstacles']
        
        # Update CONFIG with map-specific start and goal positions
        CONFIG['start_pos'] = selected_map['start_pos'].copy()
        CONFIG['goal_pos'] = selected_map['goal_pos'].copy()
        
        self.current_map_info = {
            'obstacle_count': selected_map['obstacle_count'],
            'map_id': selected_map['map_id'],
            'start_pos': selected_map['start_pos'].copy(),
            'goal_pos': selected_map['goal_pos'].copy()
        }
        
        # Create XML with selected obstacles and positions
        EnvironmentGenerator.create_xml_with_obstacles(self.obstacles)
    
    def get_curriculum_info(self):
        """Get current curriculum information"""
        if not self.curriculum_learning:
            return None
            
        return {
            'current_level': self.current_obstacle_level,
            'current_map_info': getattr(self, 'current_map_info', None),
            'total_levels': 10
        }

    def _get_obs(self):
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        goal_dist = CONFIG['goal_pos'] - pos
        
        # Get LIDAR readings (normalized to [0, 1])
        lidar_readings = self._get_lidar_readings(pos)
        
        # === LIDAR FEATURE ENGINEERING ===
        # Extract meaningful features from raw LIDAR data
        
        # 1. Minimum distance (closest obstacle)
        min_lidar = np.min(lidar_readings)
        
        # 2. Mean distance (overall clearance)
        mean_lidar = np.mean(lidar_readings)
        
        # 3. Direction to closest obstacle (unit vector)
        closest_idx = np.argmin(lidar_readings)
        closest_angle = (2 * np.pi * closest_idx) / CONFIG['lidar_num_rays']
        obstacle_direction = np.array([np.cos(closest_angle), np.sin(closest_angle)])
        
        # 4. Danger level (ratio of close obstacles)
        danger_threshold = 0.36  # 1.0m / 2.8m normalized
        num_close_obstacles = np.sum(lidar_readings < danger_threshold)
        danger_level = num_close_obstacles / CONFIG['lidar_num_rays']
        
        # 5. Directional clearance (front/right/back/left sectors)
        sector_size = len(lidar_readings) // 4
        front_clear = np.mean(lidar_readings[0:sector_size])
        right_clear = np.mean(lidar_readings[sector_size:2*sector_size])
        back_clear = np.mean(lidar_readings[2*sector_size:3*sector_size])
        left_clear = np.mean(lidar_readings[3*sector_size:4*sector_size])
        
        # 6. Goal direction alignment (unit vector toward goal)
        goal_direction_norm = goal_dist[:2] / (np.linalg.norm(goal_dist[:2]) + 1e-8)
        
        # Combine all LIDAR features (11 dimensions)
        lidar_features = np.array([
            min_lidar,                    # 1D
            mean_lidar,                   # 1D
            obstacle_direction[0],        # 1D
            obstacle_direction[1],        # 1D
            danger_level,                 # 1D
            front_clear,                  # 1D
            right_clear,                  # 1D
            back_clear,                   # 1D
            left_clear,                   # 1D
            goal_direction_norm[0],       # 1D
            goal_direction_norm[1]        # 1D
        ])
        
        return np.concatenate([pos, vel, goal_dist, lidar_readings, lidar_features])

    def _get_lidar_readings(self, pos):
        """Generate LIDAR readings in 360 degrees around the UAV (normalized to [0,1])"""
        lidar_readings = []
        
        # Define obstacle detection threshold (distance at which we consider obstacle "detected")
        obstacle_detection_threshold = 1.5  # meters
        
        for i in range(CONFIG['lidar_num_rays']):
            # Calculate ray direction (360 degrees divided by number of rays)
            angle = (2 * math.pi * i) / CONFIG['lidar_num_rays']
            ray_dir = np.array([math.cos(angle), math.sin(angle), 0])
            
            # Cast ray and find closest obstacle or boundary
            min_distance = CONFIG['lidar_range']
            closest_obstacle = None
            
            # Check boundary intersection
            boundary_dist = self._ray_boundary_intersection(pos, ray_dir)
            if boundary_dist < min_distance:
                min_distance = boundary_dist
            
            # Check obstacle intersections
            for obs in self.obstacles:
                obs_dist = self._ray_obstacle_intersection(pos, ray_dir, obs)
                if obs_dist < min_distance:
                    min_distance = obs_dist
                    closest_obstacle = obs
            
            # Log obstacle detection if obstacle is within detection threshold
            if (closest_obstacle is not None and 
                min_distance < obstacle_detection_threshold and 
                min_distance < CONFIG['lidar_range']):
                
                # Get current velocity from the data
                current_vel = self.data.qvel[:3].copy()
                
                # Log the obstacle detection
                self.log_obstacle_detection(
                    uav_pos=pos,
                    obstacle_pos=np.array(closest_obstacle['pos']),
                    obstacle_type=closest_obstacle['shape'],
                    obstacle_id=closest_obstacle['id'],
                    detection_distance=min_distance,
                    detection_angle=np.degrees(angle),
                    prev_vel=self.prev_velocity.copy(),
                    new_vel=current_vel
                )
            
            # Normalize LIDAR reading to [0, 1] range
            normalized_distance = min_distance / CONFIG['lidar_range']
            lidar_readings.append(normalized_distance)
        
        return np.array(lidar_readings)

    def _ray_boundary_intersection(self, pos, ray_dir):
        """Calculate intersection of ray with world boundaries"""
        half_world = CONFIG['world_size'] / 2
        min_dist = CONFIG['lidar_range']
        
        # Check intersection with each boundary
        boundaries = [
            (half_world, np.array([1, 0, 0])),   # +X boundary
            (-half_world, np.array([-1, 0, 0])), # -X boundary
            (half_world, np.array([0, 1, 0])),   # +Y boundary
            (-half_world, np.array([0, -1, 0]))  # -Y boundary
        ]
        
        for boundary_pos, boundary_normal in boundaries:
            # Ray-plane intersection
            denominator = np.dot(ray_dir, boundary_normal)
            if abs(denominator) > 1e-6:  # Ray not parallel to boundary
                if boundary_normal[0] != 0:  # X boundary
                    t = (boundary_pos - pos[0]) / ray_dir[0]
                else:  # Y boundary
                    t = (boundary_pos - pos[1]) / ray_dir[1]
                
                if t > 0:  # Ray goes forward
                    intersection = pos + t * ray_dir
                    # Check if intersection is within boundary limits
                    if (-half_world <= intersection[0] <= half_world and 
                        -half_world <= intersection[1] <= half_world):
                        min_dist = min(min_dist, t)
        
        return min_dist

    def _ray_obstacle_intersection(self, pos, ray_dir, obstacle):
        """Calculate intersection of ray with obstacle"""
        obs_pos = np.array(obstacle['pos'])
        min_dist = CONFIG['lidar_range']
        
        if obstacle['shape'] == 'box':
            # Simple box intersection (approximate)
            # Calculate distance to box center and subtract box size
            to_obs = obs_pos - pos
            proj_length = np.dot(to_obs, ray_dir)
            
            if proj_length > 0:  # Obstacle is in ray direction
                closest_point = pos + proj_length * ray_dir
                # Check if ray passes near the obstacle
                lateral_dist = np.linalg.norm((obs_pos - closest_point)[:2])  # Only X,Y
                
                if lateral_dist < max(obstacle['size'][0], obstacle['size'][1]):
                    # Approximate distance to obstacle surface
                    surface_dist = max(0, proj_length - max(obstacle['size'][0], obstacle['size'][1]))
                    min_dist = min(min_dist, surface_dist)
        
        elif obstacle['shape'] == 'cylinder':
            # Ray-cylinder intersection (simplified)
            to_obs = obs_pos - pos
            proj_length = np.dot(to_obs, ray_dir)
            
            if proj_length > 0:  # Obstacle is in ray direction
                closest_point = pos + proj_length * ray_dir
                lateral_dist = np.linalg.norm((obs_pos - closest_point)[:2])  # Only X,Y
                
                if lateral_dist < obstacle['size'][0]:  # Within cylinder radius
                    surface_dist = max(0, proj_length - obstacle['size'][0])
                    min_dist = min(min_dist, surface_dist)
        
        return min_dist

    def _get_reward_and_termination_info(self, obs):
        pos = obs[:3]
        vel = obs[3:6]
        goal_dist = np.linalg.norm(CONFIG['goal_pos'] - pos)
        
        # Extract LIDAR readings (normalized, indices 9:25)
        lidar_readings = obs[9:25]
        min_obstacle_dist_norm = np.min(lidar_readings)
        min_obstacle_dist = min_obstacle_dist_norm * CONFIG['lidar_range']  # Convert back to meters
        
        # Initialize termination info
        termination_info = {
            'terminated': False,
            'termination_reason': 'none',
            'final_position': pos.copy(),
            'goal_distance': goal_dist,
            'collision_detected': False,
            'out_of_bounds': False,
            'episode_length': self.step_count,
            'final_velocity': np.linalg.norm(vel[:2])  # Only horizontal velocity
        }
        
        # Initialize reward
        reward = 0.0
        
        # === TERMINAL REWARDS (Strong Signals) ===
        
        # Check if UAV is out of bounds
        if self._check_out_of_bounds(pos):
            reward = -100  # Strong penalty
            termination_info['terminated'] = True
            termination_info['termination_reason'] = 'out_of_bounds'
            termination_info['out_of_bounds'] = True
            return reward, termination_info
        
        # Check for collision with obstacles
        if self._check_collision(pos):
            reward = -100  # Strong penalty
            termination_info['terminated'] = True
            termination_info['termination_reason'] = 'collision'
            termination_info['collision_detected'] = True
            return reward, termination_info
            
        # Simple goal achievement - immediate reward and termination
        if goal_dist < 0.5:
            reward = 100  # Simple goal reward
            termination_info['terminated'] = True
            termination_info['termination_reason'] = 'goal_reached'
            print(f"✅ GOAL REACHED! Distance: {goal_dist:.3f}m")
            return reward, termination_info
        
        # === STEP REWARDS (Simplified) ===
        
        # 1. Progress reward (primary driving force)
        progress = self.prev_goal_dist - goal_dist
        reward = 10.0 * progress  # Scaled up for significance
        
        # 2. Collision avoidance rewards (positive for safe navigation)
        if min_obstacle_dist < 0.3:
            reward -= 5.0  # Danger zone - penalty for being too close
        elif min_obstacle_dist < 0.5:
            reward -= 1.0  # Warning zone - mild penalty
        elif min_obstacle_dist > 1.5:
            reward += 2.0  # Safe zone - good collision avoidance
        elif min_obstacle_dist > 1.0:
            reward += 0.5  # Moderate safety - small bonus
        
        # Update previous goal distance for next step
        self.prev_goal_dist = goal_dist
        
        return reward, termination_info

    def _check_out_of_bounds(self, pos):
        """Check if UAV position is outside the world boundaries"""
        half_world = CONFIG['world_size'] / 2
        boundary_tolerance = 0.05  # REDUCED: Stricter tolerance for numerical errors
        
        # Create a smaller effective boundary to prevent getting too close to edges
        effective_boundary = half_world - 0.1
        
        # Apply strictest check to western boundary (x negative) where we have problems
        western_boundary = -half_world + 0.2  # Even stricter western boundary
        
        return (pos[0] < western_boundary or  # Stricter western boundary
                pos[0] > effective_boundary or 
                abs(pos[1]) > effective_boundary or 
                pos[2] < 0.5 or   # Too low (but shouldn't happen with constant height)
                pos[2] > 1.5)     # Too high (but shouldn't happen with constant height)

    def _get_collision_threshold(self):
        """Get adaptive collision threshold based on training progress (curriculum)"""
        episode = self.current_episode
        
        if episode < 500:
            return 0.15  # Lenient early training
        elif episode < 1500:
            return 0.13  # Moderate
        elif episode < 3000:
            return 0.11  # Getting stricter
        else:
            return 0.10  # Final strict threshold

    def _check_collision(self, uav_pos):
        collision_threshold = self._get_collision_threshold()
        
        for obs in self.obstacles:
            obs_pos = np.array(obs['pos'])
            if obs['shape'] == 'box':
                dx = max(abs(uav_pos[0] - obs_pos[0]) - obs['size'][0], 0)
                dy = max(abs(uav_pos[1] - obs_pos[1]) - obs['size'][1], 0)
                dz = max(abs(uav_pos[2] - obs_pos[2]) - obs['size'][2], 0)
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            elif obs['shape'] == 'cylinder':
                horizontal_dist = math.sqrt((uav_pos[0]-obs_pos[0])**2 + (uav_pos[1]-obs_pos[1])**2)
                vertical_dist = abs(uav_pos[2] - obs_pos[2])
                distance = max(horizontal_dist - obs['size'][0], vertical_dist - obs['size'][1])
            
            if distance < collision_threshold:
                return True
        return False

if __name__ == '__main__':
    # Example of using the environment
    env = UAVEnv(render_mode="human")
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            done = True
    env.close()

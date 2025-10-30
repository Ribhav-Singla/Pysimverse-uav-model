# Complete Reward Structure Analysis - UAV Navigation Environment

## Overview
The UAV environment implements a **multi-component reward structure** designed to guide the agent toward safe navigation to the goal. The reward function combines terminal rewards (high-impact), step rewards (continuous feedback), and neurosymbolic balancing rewards (to help pure RL learn like symbolic rules).

---

## 1. TERMINAL REWARDS (Immediate Game-Ending Events)

### 1.1 Goal Reached ✅
- **Reward Value**: `+100`
- **Trigger**: UAV reaches goal position (distance < 0.1m)
- **Effect**: Episode terminates immediately
- **Code Location**: `_get_reward_and_termination_info()` line ~1255
- **Impact**: Strong positive signal to encourage goal-seeking behavior

### 1.2 Collision with Obstacles ⚠️
- **Reward Value**: `-100`
- **Trigger**: UAV collides with any obstacle (detected by AABB/sphere collision)
- **Collision Detection**: 
  - **Adaptive collision threshold**: Default 0.1m (can vary by curriculum)
  - **Shape support**: Boxes, cylinders, spheres
  - **3D collision checking**: Horizontal and vertical bounds validated
- **Effect**: Episode terminates immediately
- **Code Location**: `_check_collision()` method
- **Impact**: Strong negative signal to enforce obstacle avoidance

### 1.3 Out of Bounds 🚫
- **Reward Value**: `-100`
- **Triggers**: 
  - Position beyond world boundaries: x,y ∈ [-4.0, 4.0]m with 0.05m safety margin
  - Altitude violation: z < 0.3m (below minimum) or z > 2.0m (above maximum)
- **Effect**: Episode terminates immediately
- **Code Location**: `_check_out_of_bounds()` method
- **Impact**: Constrains agent to valid flight region

---

## 2. STEP REWARDS (Per-Timestep Continuous Feedback)

Applied every step (0.05s) and accumulated throughout the episode. These are the driving forces for learning behavioral patterns.

### 2.1 Progress Reward (Primary Driving Force) ⬆️
- **Formula**: `reward = 10.0 × (previous_goal_distance - current_goal_distance)`
- **Range**: 
  - **Positive**: When agent moves closer to goal (up to +10.0 per 1m progress)
  - **Negative**: When agent moves away from goal (down to -10.0 per 1m)
- **Scale**: 10x multiplier for significance
- **Update**: `self.prev_goal_dist` tracked each step
- **Code Location**: Line ~1284
- **Purpose**: Primary incentive to navigate toward goal

**Example:**
```
If UAV moves 0.5m closer to goal:
  progress = 0.5m
  reward += 10.0 × 0.5 = +5.0
```

### 2.2 Step Penalty / Living Cost ⏱️
- **Reward Value**: `CONFIG['step_reward'] = -0.2` per step
- **Configuration**: Defined at line 35 in CONFIG
- **Effect**: Penalty applied every timestep
- **Code Location**: Line ~1286
- **Purpose**: Encourages efficiency; agent must complete tasks quickly rather than wandering indefinitely
- **Impact on Episode**: 
  - 100 steps = -20 reward penalty
  - Motivates shorter episode length

**Example:**
```
Per timestep:
  reward += -0.2
  
For 500-step episode:
  Total living cost = -0.2 × 500 = -100 reward
```

### 2.3 Collision Avoidance Rewards (Proximity-Based) 🛡️
Multi-tier system based on obstacle distance from LIDAR:

#### Danger Zone (< 0.3m)
- **Reward**: `-5.0` penalty
- **Trigger**: `min_obstacle_dist < 0.3`
- **Code**: Line ~1289
- **Purpose**: Discourage flying too close to obstacles

#### Warning Zone (0.3m - 0.5m)
- **Reward**: `-1.0` penalty
- **Trigger**: `min_obstacle_dist < 0.5` (but not in danger zone)
- **Code**: Line ~1290
- **Purpose**: Mild warning to gradually maintain safe distance

**Multi-tier logic:**
```python
if min_obstacle_dist < 0.3:
    reward -= 5.0  # Danger zone
elif min_obstacle_dist < 0.5:
    reward -= 1.0  # Warning zone
# else: no penalty (safe)
```

---

## 3. NEUROSYMBOLIC BALANCING REWARDS (Neural Learning Alignment)

These rewards help the pure RL agent (λ=0) learn behavioral patterns similar to the Ripple Down Rules (RDR) symbolic system without explicitly using rules.

### 3.1 Boundary Safety Penalty 🏞️
- **Reward Value**: `-1.0` static penalty
- **Trigger**: `distance_to_boundary < 0.8m`
- **Boundary Definition**: World size ±4.0m
- **Code Location**: Line ~1295
- **Purpose**: Mirrors the `R2_BOUNDARY_SAFETY` RDR rule
- **Calculation of boundary distance**:
  ```python
  distances = [x+4.0, 4.0-x, y+4.0, 4.0-y]
  distance_to_boundary = min(distances)
  ```

**Example:**
```
At position (3.4, -3.5, 1.0):
  - Distance to east boundary: 4.0 - 3.4 = 0.6m (< 0.8m)
  - Reward += -1.0
```

### 3.2 LIDAR Goal Detection Bonus 🎯
- **Baseline Bonus**: `15.0 × max(0, goal_alignment)` per step
- **Trigger**: Goal is detectable via LIDAR (clear line of sight)
- **Code Location**: Lines ~1300-1310
- **Purpose**: Matches `R1_CLEAR_PATH` symbolic rule behavior
- **Calculation Steps**:

#### Step 1: Check if Goal Detected by LIDAR
```python
# Ray toward goal is clear if > 0.4 normalized distance
if lidar_readings[ray_index] > 0.4:
    # Check neighbors too (robustness)
    neighbors_clear = (lidar_readings[neighbor_1] > 0.3 
                      or lidar_readings[neighbor_2] > 0.3)
```

#### Step 2: Calculate Velocity-Goal Alignment
```python
current_vel = vel[:2] / norm(vel)  # Normalized 2D velocity
goal_direction = (goal_pos - pos) / norm(goal_pos - pos)
goal_alignment = dot(current_vel, goal_direction)  # Range: [-1, 1]
```

#### Step 3: Apply Bonus
```python
lidar_goal_bonus = 15.0 * max(0, goal_alignment)
reward += lidar_goal_bonus
```

**Example Scenarios:**
```
Scenario A: Goal detected, moving perfectly toward it
  - goal_alignment = +1.0
  - bonus = 15.0 × 1.0 = +15.0
  
Scenario B: Goal detected, moving perpendicular to it
  - goal_alignment = 0.0
  - bonus = 15.0 × 0.0 = +0.0 (no bonus, no penalty)
  
Scenario C: Goal detected, moving away from it
  - goal_alignment = -0.5
  - bonus = 15.0 × max(0, -0.5) = +0.0 (penalty avoided via max)
```

---

## 4. REWARD STRUCTURE TIMELINE PER EPISODE

### Example Trajectory: 200-step episode to goal

```
Step 1: Starting position [-3, -3, 1], Goal [3, 3, 1]
  Initial distance = √((3-(-3))² + (3-(-3))²) = √72 ≈ 8.49m
  
Step 1: Move 0.3m closer
  - Progress reward: 10.0 × 0.3 = +3.0
  - Living penalty: -0.2
  - No obstacle threat, no boundary issue
  - Subtotal: +2.8
  - Cumulative: +2.8
  
Step 50: Moving well, distance = 5.0m
  - Progress reward: 10.0 × 0.15 = +1.5
  - Living penalty: -0.2
  - Close to boundary (0.6m away): -1.0
  - LIDAR detects goal, velocity aligned: +12.0
  - Subtotal: +12.3
  - Cumulative: ~200 (running sum)
  
Step 100: Distance = 2.0m, clear path
  - Progress reward: 10.0 × 0.2 = +2.0
  - Living penalty: -0.2
  - Good distance from obstacles
  - LIDAR goal bonus: +8.0
  - Subtotal: +9.8
  - Cumulative: ~400
  
Step 200: Reach goal (distance < 0.1m)
  - GOAL REACHED: +100
  - Episode terminates
  - FINAL REWARD: ~500 (approximate)
```

---

## 5. REWARD COMPONENT SUMMARY TABLE

| Component | Value | Trigger | Frequency | Type |
|-----------|-------|---------|-----------|------|
| Goal Reached | +100 | dist < 0.1m | Once/episode | Terminal |
| Collision | -100 | Contact detected | Once/episode | Terminal |
| Out of Bounds | -100 | Beyond limits | Once/episode | Terminal |
| Progress | ±10.0/m | Distance change | Every step | Continuous |
| Living Penalty | -0.2 | Every step | Every step | Continuous |
| Danger Zone | -5.0 | Obstacle < 0.3m | When triggered | Continuous |
| Warning Zone | -1.0 | Obstacle 0.3-0.5m | When triggered | Continuous |
| Boundary Safety | -1.0 | Dist < 0.8m | When triggered | Continuous |
| LIDAR Goal Bonus | 0-15.0 | Goal visible + aligned | When available | Continuous |

---

## 6. REWARD SCALING & NORMALIZATION

### Raw Reward Statistics (Empirical)
- **Successful Episode**: 50-200 reward (varies by path length)
- **Failed Episode (collision)**: -20 to -100 (varies by time to collision)
- **Typical Episode Length**: 100-500 steps
- **Expected Range**: [-100, +100] with good training

### Reward per Step Characteristics
- **Minimum (failure scenarios)**: -5.2/step (danger zone + living penalty)
- **Maximum (ideal scenarios)**: +27.8/step (progress + goal bonus + boundary clear)
- **Typical**: +0 to +5/step (mixed conditions)

---

## 7. CURRICULUM IMPACT ON REWARDS

The curriculum learning affects the collision threshold but not the reward values:

### Curriculum Levels
- **Episodes 0-500**: 0.15m collision threshold (lenient)
- **Episodes 500-1500**: 0.13m collision threshold
- **Episodes 1500-3000**: 0.11m collision threshold
- **Episodes 3000+**: 0.10m collision threshold (strict)

Harder thresholds mean:
- Easier to trigger collision penalty
- Agent learns finer obstacle avoidance
- Rewards remain same, but challenges increase

---

## 8. NEUROSYMBOLIC CONTROL (Lambda Parameter)

### Lambda (λ) Values
- **λ = 0.0**: Pure RL (neurosymbolic rewards only, no RDR rules)
- **0.0 < λ < 1.0**: Hybrid mode (balanced blend)
- **λ ≥ 1.0**: RDR-first when available (use symbolic rules, fall back to RL)

### Reward Structure with λ
```
if use_neurosymbolic and λ > 0:
    # RDR system can modify actions (indirectly affects experience)
    # Neurosymbolic balancing rewards help pure RL learn rule-like behavior
else:
    # Pure RL relies entirely on environment rewards
```

The reward function itself remains identical regardless of λ. The λ parameter controls:
1. Whether RDR rules are applied to modify actions
2. What training logs are generated
3. Model checkpoint naming (for comparison)

---

## 9. CODE REFERENCE

### Main Reward Calculation Function
- **File**: `uav_env.py`
- **Function**: `_get_reward_and_termination_info(obs)` (lines 1203-1330)
- **Called from**: `step()` method (line 519)

### Configuration Constants
- **File**: `uav_env.py`
- **Location**: `CONFIG` dictionary (lines 18-36)
- **Key constants**:
  - `step_reward = -0.2`: Living penalty
  - `lidar_range = 2.9`: Detection range for obstacles
  - `collision_distance = 0.1`: Collision threshold
  - `world_size = 8.0`: ±4m boundaries

### Training Script
- **File**: `training.py`
- **Relevant section**: Lines 230-240 (reward structure documentation)
- **Lambda configuration**: Lines 53-66 (neurosymbolic config)

---

## 10. REWARD DESIGN PRINCIPLES

1. **Terminal Rewards Are Strong**: ±100 reward dominates single-step rewards
2. **Progress is Primary**: 10x multiplier ensures progress drives learning
3. **Safety is Secondary**: Proximity penalties guide safe navigation
4. **Efficiency Matters**: Living penalty encourages quick solutions
5. **Neurosymbolic Bridge**: Goal bonus helps pure RL mimic symbolic behavior
6. **Curriculum-Friendly**: Fixed reward values work with adaptive collision thresholds

---

## 11. EXPECTED LEARNING DYNAMICS

### Early Training (Episodes 0-100)
- Frequent collisions: -100 rewards dominate
- Living cost accumulates quickly
- Agent learns basic obstacle avoidance
- Progress rewards become more frequent

### Mid Training (Episodes 100-1000)
- Collisions decrease, successful navigations increase
- Progress rewards become significant (+30-50 per episode)
- Agent learns to balance speed (progress) vs safety
- Boundary safety kicks in, refining path

### Late Training (Episodes 1000+)
- Most episodes successful (>80% success rate)
- Rewards typically 50-100+ per episode
- Agent exploits LIDAR goal bonus
- Behavioral patterns stabilize

---

## 12. POTENTIAL IMPROVEMENTS (Not Implemented)

1. **Shaped Rewards**: Could add intermediate milestones
2. **Velocity Bonus**: Could reward maintaining high speed in safe conditions
3. **Energy Cost**: Could penalize acceleration/deceleration
4. **Explore Bonus**: Could encourage visiting unexplored regions early
5. **Dense Rewards**: Could reward approaching intermediate waypoints

---

## Summary

The reward structure is **well-designed for hierarchical learning**:
- **Top tier**: Terminal rewards (goal/collision) provide clear objectives
- **Middle tier**: Step rewards (progress/living cost) guide navigation
- **Bottom tier**: Proximity rewards (collision avoidance/boundary safety) ensure safe behavior
- **Enhancement layer**: Neurosymbolic bonuses help pure RL learn symbolic behaviors

This multi-layered approach enables the agent to learn complex navigation strategies while maintaining safety constraints.

# PPO Variants Implementation Guide 🚁

## Overview

This implementation provides three distinct PPO training variants for UAV navigation, each with different reward structures and behaviors:

1. **Vanilla PPO** - Basic reinforcement learning with minimal rewards
2. **AR PPO** - Augmented Rewards with additional boundary and goal detection bonuses  
3. **NS PPO** - Neurosymbolic approach with RDR rule-based assistance

---

## 🚀 Quick Start

### Command Line Usage

```bash
# Train Vanilla PPO (basic rewards only)
python training.py --ppo_type vanilla --episodes 300

# Train AR PPO (augmented rewards)  
python training.py --ppo_type ar --episodes 300

# Train NS PPO (neurosymbolic)
python training.py --ppo_type ns --episodes 300
```

### Model Files Generated

Each variant saves to a specific filename:
- `Vanilla_PPO_UAV_Weights.pth` 
- `AR_PPO_UAV_Weights.pth`
- `NS_PPO_UAV_Weights.pth`

---

## 📊 PPO Variant Details

### 1. Vanilla PPO (`--ppo_type vanilla`)

**Configuration:**
- Lambda (λ): 0.0
- Extra rewards: DISABLED
- Neurosymbolic: DISABLED

**Reward Structure:**
- ✅ Goal reached: +100
- ❌ Collision/Out of bounds: -100  
- 📈 Progress reward: 10.0 × (distance_improvement)
- ⏱️ Step penalty: -0.01 per step
- 🛡️ Basic collision avoidance: -5.0 (danger zone), -1.0 (warning zone)

**Use Case:** Baseline PPO performance testing

---

### 2. AR PPO (`--ppo_type ar`)

**Configuration:**
- Lambda (λ): 0.0
- Extra rewards: ENABLED
- Neurosymbolic: DISABLED

**Reward Structure:** 
All Vanilla PPO rewards **PLUS:**
- 🚫 **Boundary approach penalty**: -1.0 per step when within 1m of boundary
- 🎯 **LIDAR goal detection**: 1.5× reward when moving toward LIDAR-detected goal

**Boundary Penalty Details:**
- Triggers when UAV is within 1.0m of any boundary (east, west, north, south)
- Penalty scales linearly: `penalty = -1.0 × (1.0 - distance_to_boundary) / 1.0`
- Encourages UAV to stay away from world boundaries

**LIDAR Goal Detection Details:**
- Activates when goal is within LIDAR range (2.9m)
- Checks if goal direction is clear (no obstacles blocking path)
- Rewards movement toward detected goal: `reward = 1.5 × velocity_toward_goal / goal_distance`
- Only applies when UAV velocity has positive component toward goal

**Use Case:** Enhanced learning with safety and goal-seeking behaviors

---

### 3. NS PPO (`--ppo_type ns`)

**Configuration:**
- Lambda (λ): 1.0  
- Extra rewards: DISABLED
- Neurosymbolic: ENABLED

**Reward Structure:**
- Uses Vanilla PPO reward structure
- **RDR system active**: When specific rules trigger, symbolic actions override PPO actions
- Rule-based navigation assistance for obstacle avoidance and goal-seeking

**RDR System:**
- 3 built-in rules: Default, Clear Path, Boundary Safety
- Binary mode: λ=1.0 means use RDR when available, else use PPO
- Provides intelligent navigation assistance in complex scenarios

**Use Case:** Hybrid symbolic-neural approach for robust navigation

---

## 🔧 Implementation Details

### Reward Function Architecture

The reward system uses a modular approach in `_get_reward_and_termination_info()`:

```python
# Core rewards (all variants)
reward = 10.0 * progress + CONFIG['step_reward']

# AR PPO specific additions
if use_extra_rewards:
    reward += self._get_boundary_approach_penalty(pos)
    reward += self._get_lidar_goal_detection_reward(pos, vel, lidar_readings)
```

### Boundary Approach Detection

```python
def _get_boundary_approach_penalty(self, pos):
    half_world = CONFIG['world_size'] / 2  # 4.0m
    boundary_threshold = 1.0  # Penalty zone
    
    distances_to_boundaries = [
        half_world - pos[0],  # East boundary
        pos[0] + half_world,  # West boundary  
        half_world - pos[1],  # North boundary
        pos[1] + half_world   # South boundary
    ]
    
    min_distance = min(distances_to_boundaries)
    if min_distance < boundary_threshold:
        return -1.0 * (boundary_threshold - min_distance) / boundary_threshold
    return 0.0
```

### LIDAR Goal Detection

```python  
def _get_lidar_goal_detection_reward(self, pos, vel, lidar_readings):
    goal_vector = CONFIG['goal_pos'] - pos
    goal_distance = np.linalg.norm(goal_vector)
    
    if goal_distance > CONFIG['lidar_range']:  # 2.9m
        return 0.0
        
    # Check if goal direction is clear via LIDAR
    goal_angle = np.arctan2(goal_vector[1], goal_vector[0])
    ray_index = int((goal_angle / (2 * np.pi)) * 16) % 16
    goal_lidar_reading = lidar_readings[ray_index] * CONFIG['lidar_range']
    
    # Goal detected if LIDAR reading >= 90% of goal distance
    if goal_lidar_reading >= goal_distance * 0.9:
        velocity_toward_goal = np.dot(vel[:2], goal_vector[:2]) / np.linalg.norm(goal_vector[:2])
        if velocity_toward_goal > 0:
            return 1.5 * velocity_toward_goal / goal_distance
    return 0.0
```

---

## 📈 Training Progress Visualization

Each variant generates training plots with PPO-type-specific naming:
- `training_live_vanilla_ppo.png`
- `training_live_ar_ppo.png` 
- `training_live_ns_ppo.png`

Plots update every 10 episodes showing:
- Episode rewards with exponential moving average
- Success rate tracking
- Real-time performance indicators

---

## 🧪 Testing & Verification

Run the test suite to verify all variants work correctly:

```bash
# Test specific variant
python test_ppo_variants.py --variant vanilla
python test_ppo_variants.py --variant ar  
python test_ppo_variants.py --variant ns

# Test all variants
python test_ppo_variants.py --variant all
```

**Expected Test Results:**
- ✅ Environment initialization 
- ✅ Observation/action space compatibility
- ✅ Reward function execution
- ✅ AR PPO boundary penalty calculation
- ✅ AR PPO LIDAR goal detection

---

## 📋 Training Comparison

| Feature | Vanilla PPO | AR PPO | NS PPO |
|---------|-------------|--------|--------|
| **Lambda (λ)** | 0.0 | 0.0 | 1.0 |
| **Base Rewards** | ✅ | ✅ | ✅ |
| **Boundary Penalty** | ❌ | ✅ | ❌ |
| **Goal Detection** | ❌ | ✅ | ❌ |
| **RDR System** | ❌ | ❌ | ✅ |
| **Use Case** | Baseline | Enhanced | Hybrid |

---

## 🚀 Expected Training Outcomes

### Vanilla PPO
- **Strengths**: Clean baseline, pure RL learning
- **Expected behavior**: Basic goal-seeking with collision avoidance
- **Training time**: Standard convergence

### AR PPO  
- **Strengths**: Enhanced safety, improved goal detection
- **Expected behavior**: Better boundary avoidance, faster goal convergence
- **Training time**: Potentially faster due to additional guidance

### NS PPO
- **Strengths**: Intelligent rule-based assistance
- **Expected behavior**: Robust navigation, rule-guided obstacle avoidance  
- **Training time**: May converge faster due to symbolic guidance

---

## 📝 Notes

- All variants use curriculum learning (1-10 obstacle levels)
- All variants save models after every episode 
- Boundary detection uses 1.0m threshold for AR PPO
- LIDAR goal detection uses 90% threshold for path clearance
- Model weights are saved with distinct filenames for each variant

---

## 🔄 Migration from Lambda System

**Old Command:**
```bash
python training.py --lambda 0.0  # Pure RL
python training.py --lambda 1.0  # Neurosymbolic
```

**New Commands:**
```bash  
python training.py --ppo_type vanilla  # Pure RL
python training.py --ppo_type ar       # Enhanced RL
python training.py --ppo_type ns       # Neurosymbolic
```

The new system provides:
- ✅ Three distinct training modes instead of two
- ✅ Clearer semantic naming  
- ✅ Dedicated augmented rewards variant
- ✅ Better model organization
- ✅ Enhanced reward engineering

---

*Implementation completed: November 5, 2025*
*All variants tested and verified working correctly* ✅
# Rule 2 (Boundary Safety) Simplification Summary

## Date: 27 November 2025

## Overview
Simplified and re-enabled Rule 2 (Boundary Safety) in the RDR (Ripple Down Rules) system to reduce velocity only in the specific direction/axis of nearby boundaries.

---

## Changes Made

### 1. Re-enabled Rule 2
**Previous State:** Rule 2 was commented out for testing
**Current State:** Rule 2 is now active with simplified parameters

```python
rule_boundary_safety = RDRRule(
    rule_id="R2_BOUNDARY_SAFETY",
    condition=self._condition_near_boundary,
    conclusion="Near boundary: Reduce velocity in boundary direction",
    action_params={"velocity_reduction": 0.4}
)
```

### 2. Simplified Action Logic
**Previous Approach (Complex):**
- Created escape direction vectors
- Applied speed modifiers with minimum speed constraints
- Normalized and combined multiple vectors
- ~40 lines of complex logic

**New Approach (Simple):**
- Identifies which boundary is close (X or Y axis)
- Reduces velocity ONLY in that specific axis/direction
- Only reduces velocity if moving TOWARD that boundary
- ~30 lines of clear, axis-specific logic

### 3. Key Improvements

#### A. Axis-Specific Velocity Reduction
The rule now reduces velocity only in the direction of the nearby boundary:

```python
# X-axis boundaries (east/west)
if pos[0] > half_world - 0.8:  # Near east boundary
    if modified_vel[0] > 0:  # Moving eastward
        modified_vel[0] = max(0, modified_vel[0] - 0.4)

# Y-axis boundaries (north/south)  
if pos[1] > half_world - 0.8:  # Near north boundary
    if modified_vel[1] > 0:  # Moving northward
        modified_vel[1] = max(0, modified_vel[1] - 0.4)
```

#### B. Boundary-Only Activation
The rule is guaranteed to activate only for world boundaries, NOT for obstacles:

**How it's ensured:**
- `distance_to_boundary` is calculated from UAV position relative to world edges (±half_world)
- NOT derived from LIDAR readings (which include obstacles)
- Condition checks: `distance_to_boundary < 0.8` meters from world boundary

```python
# From _prepare_rdr_context()
half_world = CONFIG['world_size'] / 2
distances_to_boundaries = np.array([
    half_world + pos[0],  # Distance to west boundary
    half_world - pos[0],  # Distance to east boundary
    half_world + pos[1],  # Distance to south boundary
    half_world - pos[1]   # Distance to north boundary
])
distance_to_boundary = float(np.min(distances_to_boundaries))
```

#### C. Direction-Aware Reduction
Only reduces velocity when moving TOWARD the boundary:
- Near east boundary + moving east (vel[0] > 0) → reduce X velocity
- Near west boundary + moving west (vel[0] < 0) → reduce X velocity
- Near north boundary + moving north (vel[1] > 0) → reduce Y velocity
- Near south boundary + moving south (vel[1] < 0) → reduce Y velocity

If moving away from or parallel to the boundary, no reduction is applied.

---

## Technical Details

### Activation Threshold
- Rule activates when within **0.8 meters** of any world boundary
- World size: 8.0 meters (±4.0 meters from center)
- Activation zone: ±3.2 meters to ±4.0 meters from center

### Velocity Reduction Amount
- **0.4 units** reduction in the boundary axis direction
- Applied only when moving toward the boundary
- Final velocity clipped to [-1.0, 1.0] range

### Rule Priority
In the RDR hierarchy:
1. **R0_DEFAULT**: Fallback to PPO
2. **R1_CLEAR_PATH**: Increase velocity toward goal (when clear path exists)
3. **R2_BOUNDARY_SAFETY**: Reduce velocity in boundary direction (when near boundary)

Rule 2 is checked before the default rule but has equal priority with Rule 1.

---

## Example Scenarios

### Scenario 1: Approaching East Boundary
```
Position: [3.5, 0.0, 1.0]  (0.5m from east boundary)
Velocity: [0.6, 0.3, 0.0]  (moving east and north)
Action: Reduce X velocity only
Result: [0.2, 0.3, 0.0]    (X reduced by 0.4)
```

### Scenario 2: Near Corner (North-East)
```
Position: [3.5, 3.5, 1.0]  (near NE corner)
Velocity: [0.5, 0.5, 0.0]  (moving NE toward corner)
Action: Reduce both X and Y velocities
Result: [0.1, 0.1, 0.0]    (both reduced by 0.4)
```

### Scenario 3: Near Boundary But Moving Away
```
Position: [3.5, 0.0, 1.0]  (0.5m from east boundary)
Velocity: [-0.6, 0.3, 0.0] (moving WEST, away from boundary)
Action: No X reduction (only Y if near north/south)
Result: [-0.6, 0.3, 0.0]   (unchanged)
```

### Scenario 4: Near Obstacle (NOT a boundary)
```
Position: [0.5, 0.5, 1.0]  (center of map, near obstacle)
Velocity: [0.6, 0.6, 0.0]
Action: Rule 2 does NOT activate (not near boundary)
Result: Handled by PPO or Rule 1 instead
```

---

## Benefits of Simplified Approach

1. **Clearer Intent**: Directly reduces velocity in boundary axis
2. **More Predictable**: No complex vector math or normalization
3. **Direction-Aware**: Only reduces when moving toward boundary
4. **Axis-Independent**: X and Y axes handled separately
5. **Obstacle-Safe**: Guaranteed to NOT interfere with obstacle avoidance
6. **Easier to Debug**: Straightforward if-then logic
7. **Better Performance**: Simpler computation, fewer edge cases

---

## Testing Recommendations

1. **Boundary Approach**: Test UAV approaching each boundary (N, S, E, W)
2. **Corner Cases**: Test approaching corners (NE, NW, SE, SW)
3. **Parallel Movement**: Test moving parallel to boundaries
4. **Away Movement**: Test moving away from boundaries
5. **Obstacle vs Boundary**: Verify rule doesn't activate near obstacles
6. **Rule Priority**: Verify Rule 1 and Rule 2 interaction

---

## Code Locations

- **Rule Definition**: `uav_env.py`, lines 109-118
- **Condition Function**: `uav_env.py`, lines 129-137
- **Action Generation**: `uav_env.py`, lines 744-773
- **Context Preparation**: `uav_env.py`, lines 704-726

---

## Future Enhancements (Optional)

1. **Adaptive Reduction**: Scale reduction based on distance to boundary
   ```python
   reduction_factor = min(0.4, (0.8 - distance) * 0.5)
   ```

2. **Velocity-Aware**: Reduce more for higher velocities
   ```python
   reduction = 0.4 * abs(modified_vel[axis])
   ```

3. **Goal-Aware**: Reduce reduction if goal is near that boundary
   ```python
   if goal_near_boundary:
       velocity_reduction *= 0.5  # Gentler reduction
   ```

---

## Conclusion

Rule 2 has been successfully simplified to:
- Reduce velocity only in the specific axis/direction of nearby boundaries
- Only activate for actual world boundaries (not obstacles)
- Only reduce velocity when moving toward the boundary
- Maintain clear, predictable behavior

The simplified rule is now active and ready for testing.

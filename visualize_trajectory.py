"""
Trajectory Visualization with RDR Rule Color-Coding
Generates top-down view of NS PPO agent trajectories showing which rule was active at each point
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from pathlib import Path

def load_trajectory_data(agent_name, obstacle_level, map_id):
    """Load trajectory data for a specific test"""
    base_path = Path("Agents") / agent_name / f"obstacles_{obstacle_level}" / f"map_{map_id}"
    
    # Load trajectory
    trajectory_file = base_path / "trajectories" / "trajectory.json"
    if not trajectory_file.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")
    
    with open(trajectory_file, 'r') as f:
        trajectory = json.load(f)
    
    # Load map metadata
    metadata_file = base_path / "map_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    return trajectory, metadata

def visualize_trajectory_with_rules(agent_name="Neurosymbolic", obstacle_level=25, map_id=1):
    """
    Create top-down visualization of trajectory with RDR rule color-coding
    
    Args:
        agent_name: Agent to visualize (default: Neurosymbolic)
        obstacle_level: Obstacle difficulty level (default: 25 for high difficulty)
        map_id: Map ID to visualize (default: 1)
    """
    
    # Load data
    try:
        trajectory, metadata = load_trajectory_data(agent_name, obstacle_level, map_id)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Set up the environment bounds
    world_size = 8.0
    half_world = world_size / 2
    ax.set_xlim(-half_world - 0.5, half_world + 0.5)
    ax.set_ylim(-half_world - 0.5, half_world + 0.5)
    ax.set_aspect('equal')
    
    # Draw ground plane
    ground = patches.Rectangle((-half_world, -half_world), world_size, world_size, 
                               facecolor='#f0f0f0', edgecolor='black', linewidth=2)
    ax.add_patch(ground)
    
    # Draw boundaries (red lines)
    boundary_width = 0.05
    # North boundary
    north = patches.Rectangle((-half_world, half_world - boundary_width), world_size, boundary_width,
                              facecolor='red', alpha=0.3, edgecolor='red', linewidth=1)
    # South boundary
    south = patches.Rectangle((-half_world, -half_world), world_size, boundary_width,
                              facecolor='red', alpha=0.3, edgecolor='red', linewidth=1)
    # East boundary
    east = patches.Rectangle((half_world - boundary_width, -half_world), boundary_width, world_size,
                             facecolor='red', alpha=0.3, edgecolor='red', linewidth=1)
    # West boundary
    west = patches.Rectangle((-half_world, -half_world), boundary_width, world_size,
                             facecolor='red', alpha=0.3, edgecolor='red', linewidth=1)
    ax.add_patch(north)
    ax.add_patch(south)
    ax.add_patch(east)
    ax.add_patch(west)
    
    # Draw obstacles
    obstacles = metadata['obstacles']
    for obs in obstacles:
        x, y = obs['pos'][0], obs['pos'][1]
        
        if obs['shape'] == 'box':
            width = obs['size'][0] * 2
            height = obs['size'][1] * 2
            obstacle_patch = patches.Rectangle((x - width/2, y - height/2), width, height,
                                              facecolor=obs['color'][:3], alpha=0.7, 
                                              edgecolor='black', linewidth=1.5)
        else:  # cylinder
            radius = obs['size'][0]
            obstacle_patch = patches.Circle((x, y), radius,
                                           facecolor=obs['color'][:3], alpha=0.7,
                                           edgecolor='black', linewidth=1.5)
        ax.add_patch(obstacle_patch)
    
    # Draw start position (green)
    start_pos = metadata['start_position']
    start_marker = patches.Circle((start_pos[0], start_pos[1]), 0.15, 
                                 facecolor='green', edgecolor='darkgreen', linewidth=2, 
                                 zorder=5, label='Start')
    ax.add_patch(start_marker)
    
    # Draw goal position (blue)
    goal_pos = metadata['goal_position']
    goal_marker = patches.Circle((goal_pos[0], goal_pos[1]), 0.15, 
                                facecolor='blue', edgecolor='darkblue', linewidth=2, 
                                zorder=5, label='Goal')
    ax.add_patch(goal_marker)
    
    # Define rule colors
    rule_colors = {
        'R0_DEFAULT': '#808080',          # Gray - Default/PPO
        'R1_CLEAR_PATH': '#00FF00',       # Green - Clear path to goal
        'R2_BOUNDARY_SAFETY': '#FF0000',  # Red - Boundary safety
        'PPO': '#FFA500',                 # Orange - Pure PPO (fallback)
        'UNKNOWN': '#000000'              # Black - Unknown
    }
    
    # Extract trajectory positions and rules
    positions = []
    rules = []
    
    for step in trajectory:
        positions.append([step['position'][0], step['position'][1]])
        
        # Determine which rule was active
        rule_used = step.get('rule_used', 'PPO')
        
        # Map rule names to our color scheme
        if rule_used in ['RDR', 'SYMBOLIC']:
            # Check if there's a specific rule ID
            rule_id = step.get('rule_id', 'UNKNOWN')
            rules.append(rule_id)
        else:
            rules.append('PPO')
    
    positions = np.array(positions)
    
    # Plot trajectory segments with color coding
    for i in range(len(positions) - 1):
        rule = rules[i]
        color = rule_colors.get(rule, rule_colors['UNKNOWN'])
        
        ax.plot([positions[i, 0], positions[i+1, 0]], 
               [positions[i, 1], positions[i+1, 1]], 
               color=color, linewidth=2.5, alpha=0.8, zorder=3)
    
    # Add current position marker (last position in trajectory)
    if len(positions) > 0:
        current_pos = patches.Circle((positions[-1, 0], positions[-1, 1]), 0.12, 
                                    facecolor='yellow', edgecolor='black', linewidth=2, 
                                    zorder=6, label='Final Position')
        ax.add_patch(current_pos)
    
    # Create custom legend for rules
    legend_elements = [
        patches.Patch(facecolor=rule_colors['R1_CLEAR_PATH'], edgecolor='black', 
                     label='R1: Clear Path to Goal'),
        patches.Patch(facecolor=rule_colors['R2_BOUNDARY_SAFETY'], edgecolor='black', 
                     label='R2: Boundary Safety'),
        patches.Patch(facecolor=rule_colors['PPO'], edgecolor='black', 
                     label='Pure PPO (Neural)'),
        patches.Patch(facecolor='green', edgecolor='darkgreen', label='Start'),
        patches.Patch(facecolor='blue', edgecolor='darkblue', label='Goal'),
        patches.Patch(facecolor='yellow', edgecolor='black', label='Final Position')
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, 
             frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    
    # Labels and title
    ax.set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    ax.set_title(f'NS PPO Trajectory Visualization - Level {obstacle_level}, Map {map_id}\n'
                f'Color-coded by Active RDR Rule', 
                fontsize=15, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add info text
    total_steps = len(positions)
    
    # Count rule usage
    rule_counts = {}
    for rule in rules:
        rule_counts[rule] = rule_counts.get(rule, 0) + 1
    
    info_text = f"Total Steps: {total_steps}\n"
    info_text += f"Obstacles: {len(obstacles)}\n\n"
    info_text += "Rule Usage:\n"
    for rule, count in sorted(rule_counts.items()):
        percentage = (count / total_steps * 100) if total_steps > 0 else 0
        rule_name = rule.replace('_', ' ')
        info_text += f"{rule_name}: {percentage:.1f}%\n"
    
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
    
    plt.tight_layout()
    
    # Save figure
    output_filename = f'trajectory_visualization_NS_level{obstacle_level}_map{map_id}.png'
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Trajectory visualization saved: {output_filename}")
    print(f"   - Agent: {agent_name}")
    print(f"   - Obstacle Level: {obstacle_level}")
    print(f"   - Map ID: {map_id}")
    print(f"   - Total Steps: {total_steps}")
    print(f"   - Rule Usage: {rule_counts}")
    
    return output_filename

def visualize_multiple_maps(agent_name="Neurosymbolic", obstacle_level=25, num_maps=5):
    """Generate visualizations for multiple maps at a given difficulty level"""
    print(f"🎨 Generating trajectory visualizations for {agent_name}...")
    print(f"   Obstacle Level: {obstacle_level}")
    print(f"   Maps: 1-{num_maps}\n")
    
    generated_files = []
    
    for map_id in range(1, num_maps + 1):
        try:
            print(f"📍 Processing Map {map_id}...")
            filename = visualize_trajectory_with_rules(agent_name, obstacle_level, map_id)
            generated_files.append(filename)
        except Exception as e:
            print(f"❌ Failed to process Map {map_id}: {e}")
    
    print(f"\n✅ Generated {len(generated_files)} visualizations")
    return generated_files

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize NS PPO trajectories with RDR rule color-coding')
    parser.add_argument('--agent', type=str, default='Neurosymbolic', 
                       help='Agent name (default: Neurosymbolic)')
    parser.add_argument('--level', type=int, default=25, 
                       help='Obstacle difficulty level (default: 25)')
    parser.add_argument('--map', type=int, default=1, 
                       help='Map ID to visualize (default: 1)')
    parser.add_argument('--multiple', action='store_true', 
                       help='Generate visualizations for all 5 maps at the specified level')
    
    args = parser.parse_args()
    
    if args.multiple:
        visualize_multiple_maps(args.agent, args.level, num_maps=5)
    else:
        visualize_trajectory_with_rules(args.agent, args.level, args.map)

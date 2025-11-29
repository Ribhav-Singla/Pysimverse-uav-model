import os
import xml.etree.ElementTree as ET

# Define the agents and obstacle counts
agents = ['AR_PPO', 'Neurosymbolic', 'Pure_Neural']
obstacle_counts = range(1, 26)  # 1 to 25
maps_per_level = 5  # 5 maps per obstacle level

# Boundary parameters (based on ground plane size 4.0)
boundary_size = 4.0
wall_thickness = 0.005  # Thinner for ground lines
wall_height = 0.002  # Very thin, just above ground

def add_boundaries_to_xml(xml_path):
    """Add boundary walls to a MuJoCo XML file."""
    try:
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find the worldbody element
        worldbody = root.find('worldbody')
        if worldbody is None:
            print(f"Warning: No worldbody found in {xml_path}")
            return False
        
        # Check if boundaries already exist and remove them
        existing_boundaries = [child for child in worldbody if child.get('name', '').startswith('boundary_')]
        if existing_boundaries:
            # Silently remove old boundaries
            for boundary in existing_boundaries:
                worldbody.remove(boundary)
        
        # Add comment for boundaries
        comment = ET.Comment(' Boundary lines on ground ')
        worldbody.append(comment)
        
        # Create boundary lines on the ground
        # North line (positive Y)
        north_wall = ET.SubElement(worldbody, 'geom')
        north_wall.set('name', 'boundary_north')
        north_wall.set('type', 'box')
        north_wall.set('size', f'{boundary_size} {wall_thickness} {wall_height}')
        north_wall.set('pos', f'0 {boundary_size} {wall_height}')
        north_wall.set('rgba', '1.0 0.0 0.0 1.0')
        
        # South line (negative Y)
        south_wall = ET.SubElement(worldbody, 'geom')
        south_wall.set('name', 'boundary_south')
        south_wall.set('type', 'box')
        south_wall.set('size', f'{boundary_size} {wall_thickness} {wall_height}')
        south_wall.set('pos', f'0 {-boundary_size} {wall_height}')
        south_wall.set('rgba', '1.0 0.0 0.0 1.0')
        
        # East line (positive X)
        east_wall = ET.SubElement(worldbody, 'geom')
        east_wall.set('name', 'boundary_east')
        east_wall.set('type', 'box')
        east_wall.set('size', f'{wall_thickness} {boundary_size} {wall_height}')
        east_wall.set('pos', f'{boundary_size} 0 {wall_height}')
        east_wall.set('rgba', '1.0 0.0 0.0 1.0')
        
        # West line (negative X)
        west_wall = ET.SubElement(worldbody, 'geom')
        west_wall.set('name', 'boundary_west')
        west_wall.set('type', 'box')
        west_wall.set('size', f'{wall_thickness} {boundary_size} {wall_height}')
        west_wall.set('pos', f'{-boundary_size} 0 {wall_height}')
        west_wall.set('rgba', '1.0 0.0 0.0 1.0')
        
        # Write the modified XML back to file
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        # Success - no print for cleaner output
        return True
        
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return False

def main():
    """Add boundaries to all map.xml files in the new folder structure."""
    agents_dir = 'Agents'
    total_processed = 0
    total_modified = 0
    total_not_found = 0
    
    print("🔧 Adding boundaries to map.xml files...")
    print(f"📊 Processing {len(agents)} agents × {len(list(obstacle_counts))} obstacle levels × {maps_per_level} maps")
    print("=" * 60)
    
    for agent in agents:
        print(f"\n🤖 Processing agent: {agent}")
        agent_processed = 0
        agent_modified = 0
        
        for obstacle_count in obstacle_counts:
            # Process all maps for this obstacle level
            for map_id in range(1, maps_per_level + 1):
                xml_path = os.path.join(agents_dir, agent, f'obstacles_{obstacle_count}', f'map_{map_id}', 'map.xml')
                
                if os.path.exists(xml_path):
                    total_processed += 1
                    agent_processed += 1
                    if add_boundaries_to_xml(xml_path):
                        total_modified += 1
                        agent_modified += 1
                else:
                    total_not_found += 1
                    # Only print if verbose mode needed
                    # print(f"File not found: {xml_path}")
        
        print(f"   ✓ {agent}: {agent_modified}/{agent_processed} files modified")
    
    print(f"\n{'=' * 60}")
    print(f"=== Summary ===")
    print(f"Total files found: {total_processed}")
    print(f"Total files modified: {total_modified}")
    print(f"Total files not found: {total_not_found}")
    print(f"Success rate: {total_modified}/{total_processed} ({100*total_modified/total_processed if total_processed > 0 else 0:.1f}%)")

if __name__ == "__main__":
    main()

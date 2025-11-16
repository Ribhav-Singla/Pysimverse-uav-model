import os
import xml.etree.ElementTree as ET

# Define the agents and obstacle counts
agents = ['AR_PPO', 'Neurosymbolic', 'Pure_Neural']
obstacle_counts = range(1, 16)  # 1 to 15

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
            print(f"Removing old boundaries from {xml_path}...")
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
        print(f"Added boundaries to {xml_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return False

def main():
    """Add boundaries to all map.xml files."""
    agents_dir = 'Agents'
    total_processed = 0
    total_modified = 0
    
    for agent in agents:
        for obstacle_count in obstacle_counts:
            xml_path = os.path.join(agents_dir, agent, f'obstacles_{obstacle_count}', 'map.xml')
            
            if os.path.exists(xml_path):
                total_processed += 1
                if add_boundaries_to_xml(xml_path):
                    total_modified += 1
            else:
                print(f"File not found: {xml_path}")
    
    print(f"\n=== Summary ===")
    print(f"Total files processed: {total_processed}")
    print(f"Total files modified: {total_modified}")
    print(f"Total files skipped: {total_processed - total_modified}")

if __name__ == "__main__":
    main()

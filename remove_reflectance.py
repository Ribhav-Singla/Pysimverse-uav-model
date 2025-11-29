import os
import xml.etree.ElementTree as ET

# Define the agents and obstacle counts
agents = ['AR_PPO', 'Neurosymbolic', 'Pure_Neural']
obstacle_counts = range(1, 26)  # 1 to 25
maps_per_level = 5  # 5 maps per obstacle level

def remove_reflectance_from_xml(xml_path):
    """Remove reflectance from the grid material in a MuJoCo XML file."""
    try:
        # Parse the XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find the asset element
        asset = root.find('asset')
        if asset is None:
            print(f"Warning: No asset found in {xml_path}")
            return False
        
        # Find the grid material
        modified = False
        for material in asset.findall('material'):
            if material.get('name') == 'grid':
                if 'reflectance' in material.attrib:
                    del material.attrib['reflectance']
                    modified = True
                    # Silently remove - cleaner output for bulk processing
        
        if modified:
            # Write the modified XML back to file
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return False

def main():
    """Remove reflectance from all map.xml files in the new folder structure."""
    agents_dir = 'Agents'
    total_processed = 0
    total_modified = 0
    total_not_found = 0
    
    print("🔧 Removing reflectance from map.xml files...")
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
                    if remove_reflectance_from_xml(xml_path):
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

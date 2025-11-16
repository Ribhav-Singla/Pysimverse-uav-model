import os
import xml.etree.ElementTree as ET

# Define the agents and obstacle counts
agents = ['AR_PPO', 'Neurosymbolic', 'Pure_Neural']
obstacle_counts = range(1, 16)  # 1 to 15

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
                    print(f"Removed reflectance from {xml_path}")
                else:
                    print(f"No reflectance found in {xml_path}")
        
        if modified:
            # Write the modified XML back to file
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return False

def main():
    """Remove reflectance from all map.xml files."""
    agents_dir = 'Agents'
    total_processed = 0
    total_modified = 0
    
    for agent in agents:
        for obstacle_count in obstacle_counts:
            xml_path = os.path.join(agents_dir, agent, f'obstacles_{obstacle_count}', 'map.xml')
            
            if os.path.exists(xml_path):
                total_processed += 1
                if remove_reflectance_from_xml(xml_path):
                    total_modified += 1
            else:
                print(f"File not found: {xml_path}")
    
    print(f"\n=== Summary ===")
    print(f"Total files processed: {total_processed}")
    print(f"Total files modified: {total_modified}")
    print(f"Total files skipped: {total_processed - total_modified}")

if __name__ == "__main__":
    main()

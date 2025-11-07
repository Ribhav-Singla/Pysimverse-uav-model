import sys
print("Python path:", sys.path)

try:
    print("Importing uav_env...")
    from uav_env import CONFIG
    print("SUCCESS: CONFIG imported")
    print("Depth range:", CONFIG.get('depth_range', 'NOT FOUND'))
    print("Depth features dim:", CONFIG.get('depth_features_dim', 'NOT FOUND'))
except Exception as e:
    print("ERROR importing uav_env:", e)
    import traceback
    traceback.print_exc()
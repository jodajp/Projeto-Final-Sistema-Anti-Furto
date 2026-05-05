import json
from pathlib import Path

# Check several files
data_dir = Path('Data')
files = list(data_dir.rglob('*.json'))[:3]

for json_file in files:
    with open(json_file) as f:
        data = json.load(f)
    
    camera = list(data.keys())[0]
    person = list(data[camera].keys())[0]
    kp = data[camera][person]['keypoints']
    
    print(f'File: {json_file.name}')
    print(f'  Keypoints length: {len(kp)} (type: {type(kp[0]).__name__})')
    is_single = len(kp) == 51 and isinstance(kp[0], float)
    print(f'  Structure: {"single 51-element array" if is_single else "list of frames"}')
    print()

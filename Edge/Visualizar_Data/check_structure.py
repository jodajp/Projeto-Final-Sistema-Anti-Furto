import json
from pathlib import Path

# Check the file from the manifest
pose_file = Path("Data/RetailS_train/pose/train/1_041812_1.json")
with open(pose_file) as f:
    data = json.load(f)

print("File structure:")
print(f"Top-level keys (cameras): {list(data.keys())}")

cam_id = list(data.keys())[0]
print(f"\nCamera {cam_id}:")
people_ids = list(data[cam_id].keys())
print(f"  Number of people: {len(people_ids)}")
print(f"  First 5 person IDs: {people_ids[:5]}")

# Check first person's data structure
person_id = people_ids[0]
person_data = data[cam_id][person_id]
print(f"\nPerson {person_id}:")
print(f"  Keys: {list(person_data.keys())}")
print(f"  Keypoints: {type(person_data['keypoints']).__name__} with {len(person_data['keypoints'])} elements")

kp = person_data['keypoints']
print(f"  First 6 values: {kp[:6]}")

# Check if this looks like a single frame (51 values = 17*3) or multiple frames
if len(kp) == 51:
    print(f"\n✓ Single frame for person (17 joints × 3 = 51 values)")
elif len(kp) % 51 == 0:
    print(f"\n⚠ Multiple frames ({len(kp) // 51} frames of 51 values each)")
else:
    print(f"\n❓ Unusual structure: {len(kp)} values (not multiple of 51)")

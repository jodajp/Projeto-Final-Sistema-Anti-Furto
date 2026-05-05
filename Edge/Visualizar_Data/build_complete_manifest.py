#!/usr/bin/env python
"""
Build complete manifests for RetailS datasets including training data.

This script creates comprehensive manifests that:
- Index the nested JSON structure correctly
- Include training data (19.9M+ normal frames)
- Map person_id to source files and frame ranges
- Support frame interpolation metadata
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent / 'Data'
TRAIN_DIR = DATA_DIR / 'RetailS_train' / 'pose' / 'train'
REALWORLD_DIR = DATA_DIR / 'RetailS_test_realworld' / 'pose' / 'test'
STAGED_DIR = DATA_DIR / 'RetailS_test_staged' / 'pose' / 'test'
OUTPUT_DIR = Path(__file__).parent / 'Manifests'

# Create output dir
OUTPUT_DIR.mkdir(exist_ok=True)


def explore_json_structure_fast(json_file):
    """
    Explore JSON file quickly - just get structure without full parsing.
    """
    try:
        with open(json_file) as f:
            # Read first part to determine structure
            data = json.load(f)
        
        people_info = {}
        for person_id, frames_dict in data.items():
            # Just count frames, don't store all indices
            frame_indices = sorted([int(k) for k in frames_dict.keys()])
            people_info[person_id] = {
                'num_frames': len(frame_indices),
                'first_frame': min(frame_indices),
                'last_frame': max(frame_indices)
            }
        
        return people_info
    except Exception as e:
        logger.warning(f"Error loading {json_file}: {e}")
        return {}


def explore_json_structure(json_file):
    """
    Explore JSON file and return info about people and frames.
    
    Returns dict mapping:
    {
        person_id: {
            "num_frames": int,
            "frame_indices": [list of frame numbers],
            "size_mb": float
        }
    }
    """
    return explore_json_structure_fast(json_file)


def build_train_manifest():
    """Build manifest for training set (normal behavior)."""
    logger.info(f"\n📊 Scanning training set: {TRAIN_DIR}")
    
    manifest = []
    person_counter = 0
    
    json_files = sorted(list(TRAIN_DIR.glob('*.json')))
    logger.info(f"   Found {len(json_files)} JSON files")
    
    for json_file in tqdm(json_files, desc="  Processing files"):
        people_info = explore_json_structure(json_file)
        
        for person_id, info in people_info.items():
            if info['num_frames'] < 2:
                continue  # Skip single frames
            
            person_counter += 1
            entry = {
                'person_id': f'train_{person_counter:06d}',
                'dataset': 'Training',
                'source_file': json_file.name,
                'original_camera': json_file.name.split('_')[0],
                'original_person_id': person_id,
                'num_frames': info['num_frames'],
                'frame_range': [info['first_frame'], info['last_frame']],
                'quality_score': info['num_frames'] * 0.5,  # Longer = better
                'label': 'normal'  # Training set is normal behavior
            }
            manifest.append(entry)
    
    logger.info(f"   ✅ Found {person_counter} people with sequences")
    return manifest


def build_staged_manifest():
    """Build manifest for staged (shoplifting) test set."""
    logger.info(f"\n📊 Scanning staged test set: {STAGED_DIR}")
    
    manifest = []
    person_counter = 0
    
    json_files = sorted(list(STAGED_DIR.glob('*.json')))
    logger.info(f"   Found {len(json_files)} JSON files")
    
    for json_file in tqdm(json_files, desc="  Processing files"):
        people_info = explore_json_structure(json_file)
        
        for person_id, info in people_info.items():
            if info['num_frames'] < 2:
                continue
            
            person_counter += 1
            entry = {
                'person_id': f'staged_{person_counter:06d}',
                'dataset': 'Staged-Test',
                'source_file': json_file.name,
                'original_camera': json_file.name.split('_')[0],
                'original_person_id': person_id,
                'num_frames': info['num_frames'],
                'frame_range': [info['first_frame'], info['last_frame']],
                'quality_score': info['num_frames'] * 0.5,
                'label': 'suspicious'  # Staged is shoplifting
            }
            manifest.append(entry)
    
    logger.info(f"   ✅ Found {person_counter} people with sequences")
    return manifest


def build_realworld_manifest():
    """Build manifest for real-world test set (mixed)."""
    logger.info(f"\n📊 Scanning real-world test set: {REALWORLD_DIR}")
    
    manifest = []
    normal_counter = 0
    suspicious_counter = 0
    
    json_files = sorted(list(REALWORLD_DIR.glob('*.json')))
    logger.info(f"   Found {len(json_files)} JSON files")
    
    for json_file in tqdm(json_files, desc="  Processing files"):
        people_info = explore_json_structure(json_file)
        
        for person_id, info in people_info.items():
            if info['num_frames'] < 2:
                continue
            
            # For real-world, we don't know labels - treat as suspicious for safety
            suspicious_counter += 1
            entry = {
                'person_id': f'realworld_suspicious_{suspicious_counter:06d}',
                'dataset': 'Real-world-Test',
                'source_file': json_file.name,
                'original_camera': json_file.name.split('_')[0],
                'original_person_id': person_id,
                'num_frames': info['num_frames'],
                'frame_range': [info['first_frame'], info['last_frame']],
                'quality_score': info['num_frames'] * 0.5,
                'label': 'suspicious'  # Real-world has some shoplifting
            }
            manifest.append(entry)
    
    logger.info(f"   ✅ Found {suspicious_counter} suspicious sequences")
    return manifest


def save_manifest(manifest, filename, category_name):
    """Save manifest to JSON file."""
    output_file = OUTPUT_DIR / filename
    
    with open(output_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"   ✅ Saved {len(manifest)} entries to {output_file.name}")
    
    # Summary stats
    normal_count = sum(1 for e in manifest if e.get('label') == 'normal')
    suspicious_count = sum(1 for e in manifest if e.get('label') == 'suspicious')
    total_frames = sum(e.get('num_frames', 0) for e in manifest)
    
    logger.info(f"\n📈 {category_name} Summary:")
    logger.info(f"   Total entries: {len(manifest)}")
    logger.info(f"   Normal: {normal_count}")
    logger.info(f"   Suspicious: {suspicious_count}")
    logger.info(f"   Total frames: {total_frames:,}")
    
    return output_file


def main():
    """Build all manifests."""
    print("\n" + "="*70)
    print("🗺️  RETAIL S MANIFEST BUILDER")
    print("="*70)
    
    # Build individual manifests
    train_manifest = build_train_manifest()
    staged_manifest = build_staged_manifest()
    realworld_manifest = build_realworld_manifest()
    
    # Save individual manifests
    print("\n" + "-"*70)
    print("💾 SAVING MANIFESTS")
    print("-"*70)
    
    save_manifest(train_manifest, 'manifest_train.json', 'Training Set')
    save_manifest(staged_manifest, 'manifest_staged.json', 'Staged Test Set')
    save_manifest(realworld_manifest, 'manifest_realworld.json', 'Real-world Test Set')
    
    # Create combined normal manifest (train + normal from realworld)
    combined_normal = train_manifest + [
        e for e in realworld_manifest if e.get('label') == 'normal'
    ]
    save_manifest(combined_normal, 'manifest_normal_combined.json', 'Combined Normal')
    
    # Create combined suspicious manifest
    combined_suspicious = staged_manifest + [
        e for e in realworld_manifest if e.get('label') == 'suspicious'
    ]
    save_manifest(combined_suspicious, 'manifest_suspicious_combined.json', 'Combined Suspicious')
    
    # Create all-in-one manifest
    all_manifest = train_manifest + staged_manifest + realworld_manifest
    save_manifest(all_manifest, 'manifest_all.json', 'All Datasets')
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"\nManifests location: {OUTPUT_DIR}/")
    print("\nFiles created:")
    print("  - manifest_train.json (training normal behavior)")
    print("  - manifest_staged.json (shoplifting events)")
    print("  - manifest_realworld.json (real-world test)")
    print("  - manifest_normal_combined.json (all normal behavior)")
    print("  - manifest_suspicious_combined.json (all suspicious)")
    print("  - manifest_all.json (everything)")


if __name__ == '__main__':
    main()

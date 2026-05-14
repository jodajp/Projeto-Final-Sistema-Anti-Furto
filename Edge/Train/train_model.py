import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add root folder to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Use sequence length of 45 frames (~1.5 sec) to capture full pocketing motion
SEQ_LENGTH = 45
KEYPOINT_DIM = 34  # 17 keypoints * 2 (x, y)

class SimpleActionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ActionClassifier(nn.Module):
    def __init__(self, seq_len=SEQ_LENGTH, input_dim=KEYPOINT_DIM, hidden_dim=64):
        super().__init__()
        # Flatten temporal sequence to fixed vector, then MLP
        # This is simple, fast, and doesn't require recurrent/graph logic!
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(seq_len * input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # Normal (0), Suspicious (1)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

def load_custom_suspicious_pkl():
    """Load sequences from the generated PKL file."""
    pkl_path = ROOT_DIR / "Visualizar_Data" / "Output" / "custom_shoplifting_dataset.pkl"
    seqs = []
    if not pkl_path.exists():
        print(f"File not found: {pkl_path}")
        return seqs
        
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    def _select_center_window(kpts):
        """Select a single window centered on the likely pocketing action.
        Heuristic: choose the frame where either wrist is closest to the hip center.
        """
        T = kpts.shape[0]
        if T == 0:
            return None

        # Indices: left_wrist=9, right_wrist=10, left_hip=11, right_hip=12
        wrists = []
        for widx in (9, 10):
            w = kpts[:, widx, :]
            wrists.append(w)
        wrists = np.stack(wrists, axis=1)  # (T, 2, 2)

        # hip center per frame (prefer hips, fallback to shoulders)
        hips = kpts[:, [11, 12], :]
        valid_hips = ~np.isnan(hips).any(axis=2)
        hip_center = np.nanmean(hips, axis=1)
        # if hip_center contains NaNs, fallback to shoulder center (5,6)
        if np.isnan(hip_center).any():
            shoulders = kpts[:, [5, 6], :]
            hip_center = np.nanmean(np.where(np.isnan(hip_center)[:, None, :], shoulders, hips), axis=1)

        # compute min wrist-to-hip distance per frame
        dists = np.full((T,), np.inf)
        for t in range(T):
            hc = hip_center[t]
            if np.isnan(hc).any():
                continue
            ws = []
            for w in range(2):
                wpt = wrists[t, w]
                if np.isnan(wpt).any():
                    continue
                ws.append(np.linalg.norm(wpt - hc))
            if ws:
                dists[t] = min(ws)

        # choose frame with minimum distance
        if np.isfinite(dists).any():
            center = int(np.nanargmin(dists))
        else:
            center = T // 2

        # build window centered on center
        start = max(0, center - SEQ_LENGTH // 2)
        end = start + SEQ_LENGTH
        if end > T:
            end = T
            start = max(0, end - SEQ_LENGTH)

        window = kpts[start:end]
        if len(window) < SEQ_LENGTH:
            window = np.pad(window, ((0, SEQ_LENGTH - len(window)), (0, 0), (0, 0)), mode='edge')
        return window

    for ann in data['annotations']:
        # keypoints shape: (1, T, 17, 2), drop first dimension
        kpts = ann['keypoint'][0]
        if kpts is None or len(kpts) == 0:
            continue
        window = _select_center_window(kpts)
        if window is None:
            continue
        flat_window = _normalize_window(window)
        seqs.append(flat_window)

    return seqs

def _normalize_window(kpts):
    """Normalize a temporal window of (SEQ_LENGTH, 17, 2) keypoints.
    Locally center to the torso or just min-max to [0,1] for MLP."""
    normalized = np.zeros_like(kpts)
    for t in range(kpts.shape[0]):
        frame_kpts = kpts[t]
        # Ignore NaNs during min/max
        valid_kpts = frame_kpts[~np.isnan(frame_kpts).any(axis=1)]
        if len(valid_kpts) > 0:
            min_xy = np.min(valid_kpts, axis=0)
            max_xy = np.max(valid_kpts, axis=0)
            range_xy = np.maximum(max_xy - min_xy, 1e-5)
            normalized[t] = (frame_kpts - min_xy) / range_xy
        else:
            normalized[t] = frame_kpts
    
    # Fill remaining NaNs
    return np.nan_to_num(normalized)

def load_retails_data(manifest_name, max_samples=500):
    """Load random sequences from RetailS dataset JSONs."""
    manifest_path = ROOT_DIR / "Visualizar_Data" / "Manifests" / manifest_name
    seqs = []
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return seqs
        
    data_dir = ROOT_DIR / "Visualizar_Data" / "Data"
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    # Shuffle for randomness
    np.random.shuffle(manifest)
    
    loaded_clips = 0
    pbar = tqdm(total=max_samples, desc=f"Loading {manifest_name}")
    
    for item in manifest:
        if loaded_clips >= max_samples:
            break
            
        dataset_name = item['dataset']
        source_file = item['source_file']
        person_id = str(item['original_person_id'])
        frame_range = item.get('frame_range', [0, 0])
        
        # Build path to json
        if dataset_name == "Training":
            json_path = data_dir / "RetailS_train" / "pose" / "train" / source_file
        elif dataset_name == "Realworld":
            json_path = data_dir / "RetailS_test_realworld" / "pose" / "test" / source_file
        elif dataset_name == "Staged":
            json_path = data_dir / "RetailS_test_staged" / "pose" / "test" / source_file
        else:
            continue
            
        if not json_path.exists():
            continue
            
        try:
            with open(json_path, 'r') as f:
                pose_data = json.load(f)
        except Exception as e:
            continue
            
        # Match person ID format properly
        # The JSON uses strings like "1", but original_person_id might be an int
        actual_person_id = None
        for pid in pose_data.keys():
            if str(pid) == person_id:
                actual_person_id = pid
                break
                
        if not actual_person_id:
            # Fallback to first person if only 1 person
            if len(pose_data) == 1:
                actual_person_id = list(pose_data.keys())[0]
            else:
                continue
            
        person_frames = pose_data[actual_person_id]
        
        # Extract a contiguous SEQ_LENGTH chunk
        start_f = frame_range[0]
        end_f = frame_range[1]
        
        if end_f - start_f < SEQ_LENGTH:
            continue
            
        # Extract keys that are numeric
        frame_keys = sorted([int(k) for k in person_frames.keys() if k.isdigit()])
        
        if not frame_keys:
            continue
            
        # Find a sub-sequence of 30 frames
        valid_start = max(start_f, frame_keys[0])
        valid_end = min(end_f, frame_keys[-1])
        
        if valid_end - valid_start < SEQ_LENGTH:
            continue
            
        # Collect full available frame range for this person within annotated range
        frames_section = []
        for f in range(valid_start, valid_end + 1):
            f_str = str(f)
            if f_str in person_frames:
                kpts_raw = np.array(person_frames[f_str]['keypoints'])
                if kpts_raw.shape == (51,):
                    kpts = kpts_raw.reshape(17, 3)[:, :2]
                elif kpts_raw.shape == (34,):
                    kpts = kpts_raw.reshape(17, 2)
                else:
                    kpts = kpts_raw
                frames_section.append(kpts)
            else:
                if frames_section:
                    frames_section.append(frames_section[-1])
                else:
                    # skip this clip if initial frames missing
                    frames_section = []
                    break

        if len(frames_section) >= SEQ_LENGTH:
            sec_kpts = np.stack(frames_section)

            # Heuristic: choose the frame where wrist is closest to hip center
            T = sec_kpts.shape[0]
            wrists = []
            for widx in (9, 10):
                wrists.append(sec_kpts[:, widx, :])
            wrists = np.stack(wrists, axis=1)
            hips = sec_kpts[:, [11, 12], :]
            hip_center = np.nanmean(hips, axis=1)
            if np.isnan(hip_center).any():
                shoulders = sec_kpts[:, [5, 6], :]
                hip_center = np.nanmean(np.where(np.isnan(hip_center)[:, None, :], shoulders, hips), axis=1)

            dists = np.full((T,), np.inf)
            for t in range(T):
                hc = hip_center[t]
                if np.isnan(hc).any():
                    continue
                ws = []
                for w in range(2):
                    wpt = wrists[t, w]
                    if np.isnan(wpt).any():
                        continue
                    ws.append(np.linalg.norm(wpt - hc))
                if ws:
                    dists[t] = min(ws)

            if np.isfinite(dists).any():
                center_idx = int(np.nanargmin(dists))
            else:
                center_idx = T // 2

            start = max(0, center_idx - SEQ_LENGTH // 2)
            end = start + SEQ_LENGTH
            if end > T:
                end = T
                start = max(0, end - SEQ_LENGTH)

            window = sec_kpts[start:end]
            if len(window) < SEQ_LENGTH:
                window = np.pad(window, ((0, SEQ_LENGTH - len(window)), (0, 0), (0, 0)), mode='edge')

            seqs.append(_normalize_window(window))
            loaded_clips += 1
            pbar.update(1)
            
    pbar.close()
    return seqs


def main():
    print("=" * 70)
    print("TRAINING ANTI-THEFT ACTION MODEL")
    print("=" * 70)
    
    # 1. Load Suspicious (Shoplifting)
    print("\n[1/4] Loading Suspicious Data...")
    suspicious_seqs = load_custom_suspicious_pkl()
    print(f"Loaded {len(suspicious_seqs)} suspicious sequences from custom PKL.")
    
    # Load more if needed
    if len(suspicious_seqs) < 200:
        print("Loading additional suspicious data from RetailS (Staged Test)...")
        extra_suspicious = load_retails_data("manifest_suspicious_combined.json", max_samples=400)
        suspicious_seqs.extend(extra_suspicious)
        
    print(f"Total Suspicious (Class 1) Sequences: {len(suspicious_seqs)}")
    
    # 2. Load Normal Data
    print("\n[2/4] Loading Normal Data...")
    num_normal_needed = max(min(len(suspicious_seqs) * 2, 2000), 200) # Ensure enough normal data
    normal_seqs = load_retails_data("manifest_normal_combined.json", max_samples=num_normal_needed)
    print(f"Total Normal (Class 0) Sequences: {len(normal_seqs)}")
    
    if len(suspicious_seqs) == 0 or len(normal_seqs) == 0:
        print("Not enough data to train. Exiting.")
        return
        
    # 3. Prepare Dataset
    print("\n[3/4] Preparing Dataset...")
    X = np.array(suspicious_seqs + normal_seqs)
    y = np.array([1]*len(suspicious_seqs) + [0]*len(normal_seqs))
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Train/Val Split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    train_dataset = SimpleActionDataset(X_train, y_train)
    val_dataset = SimpleActionDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
    
    # 4. Train Model
    print("\n[4/4] Training Model...")
    model = ActionClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 30
    best_acc = 0.0
    
    out_model_path = ROOT_DIR / "models" / "activity_classifier.pth"
    out_model_path.parent.mkdir(exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        val_acc = 100 * correct / total if total > 0 else 0

        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/max(1,len(val_loader)):.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_model_path)
            
    print(f"\n✓ Training complete! Best Validation Accuracy: {best_acc:.2f}%")
    print(f"✓ Model saved to {out_model_path}")

if __name__ == '__main__':
    main()


import numpy as np
import sys
import os

def read_ply_simple(path):
    with open(path, 'rb') as f:
        header = b""
        line = f.readline()
        num_verts = 0
        while line and b"end_header" not in line:
            header += line
            if b"element vertex" in line:
                num_verts = int(line.split()[-1])
            line = f.readline()
        data = f.read()
    if b"format binary_little_endian" in header:
        full_dt = []
        for l in header.split(b'\n'):
            if l.startswith(b"property"):
                parts = l.split()
                if parts[1] == b"float":
                    full_dt.append((parts[2].decode(), 'f4'))
                elif parts[1] == b"uchar":
                    full_dt.append((parts[2].decode(), 'u1'))
        actual_dt = np.dtype(full_dt)
        pts_data = np.frombuffer(data, dtype=actual_dt, count=num_verts)
        pts = np.vstack([pts_data['x'], pts_data['y'], pts_data['z']]).T
    else:
        pts = np.loadtxt(path, skiprows=header.count(b'\n')+1, usecols=(0,1,2))
        pts = pts[:num_verts]
    return pts

def get_median_nn(pts, sample_size=1000):
    if len(pts) < 2: return 0
    # Use a small random sample to keep it fast without specialized libraries
    idx = np.random.choice(len(pts), min(len(pts), sample_size), replace=False)
    sample = pts[idx]
    
    # Brute force NN for the sample against the whole set (or just the sample for speed)
    # Using sample vs sample for a quick estimate
    from scipy.spatial.distance import pdist, squareform
    if len(sample) > 2000: # Safety
        sample = sample[:2000]
    dists = squareform(pdist(sample))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    return np.median(min_dists)

scenes = ["Box", "Entrance", "Fountain", "Statue"]
base_path = "StemGames2026_ProjectTask/outputs"

for scene in scenes:
    path = f"{base_path}/{scene}/scene_cloud.ply"
    if not os.path.exists(path):
        print(f"{scene}: File not found")
        continue
    try:
        pts = read_ply_simple(path)
    except Exception as e:
        print(f"{scene}: Error {e}")
        continue
    
    vmin, vmax = pts.min(axis=0), pts.max(axis=0)
    extent = vmax - vmin
    
    # Try to use scipy for distance if available, otherwise just use numpy
    try:
        med_nn = get_median_nn(pts)
    except ImportError:
        # Very crude fallback
        med_nn = -1 

    print(f"Scene: {scene}")
    print(f"  Vertices: {len(pts)}")
    print(f"  Min: {vmin}")
    print(f"  Max: {vmax}")
    print(f"  Extent: {extent}")
    print(f"  Median NN distance: {med_nn:.6f}")
    print("-" * 20)

import numpy as np
from PIL import Image
import os
import re

def parse_k(k_path):
    with open(k_path, 'r') as f:
        content = f.read()
    # Extract numbers from the matrix string
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content.split('=')[-1])
    arr = np.array([float(x) for x in numbers]).reshape(3, 3)
    return arr

def compute_fov(k_path, img_dir):
    K = parse_k(k_path)
    fx = K[0, 0]
    fy = K[1, 1]
    
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        return None, None
    
    img = Image.open(os.path.join(img_dir, img_files[0]))
    w, h = img.size
    
    hfov = 2 * np.arctan(w / (2 * fx)) * 180 / np.pi
    vfov = 2 * np.arctan(h / (2 * fy)) * 180 / np.pi
    return hfov, vfov

scenes = {
    "Statue": ("StemGames2026_ProjectTask/TestImages/Statue/K.txt", "StemGames2026_ProjectTask/TestImages/Statue"),
    "Fountain": ("StemGames2026_ProjectTask/TestImages/Fountain/K.txt", "StemGames2026_ProjectTask/TestImages/Fountain")
}

for name, (k_path, img_dir) in scenes.items():
    hfov, vfov = compute_fov(k_path, img_dir)
    print(f"{name}: HFOV={hfov:.2f}, VFOV={vfov:.2f}")

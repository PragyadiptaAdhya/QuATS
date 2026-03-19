import numpy as np
from config import Config
config = Config()
def check_overlap_and_mark(cx, cy, crop_w, crop_h, occupied_mask):
    """Checks for overlap. Returns True if overlapping, False if safe to place."""
    y1, y2 = cy - crop_h // 2, cy + crop_h // 2
    x1, x2 = cx - crop_w // 2, cx + crop_w // 2
    
    if np.any(occupied_mask[y1:y2, x1:x2]):
        return True
        
    occupied_mask[y1:y2, x1:x2] = True
    return False

def extract_uniform_random_patches(img_rgb):
    
    H, W, _ = img_rgb.shape
    crop_w, crop_h = config.PATCH_SIZE, config.PATCH_SIZE
    margin_y, margin_x = crop_h // 2, crop_w // 2

    # Calculate capacity using the 1.5 scaling factor
    max_cols = int(W // (1.5 * crop_w))
    max_rows = int(H // (1.5 * crop_h))
    
    if max_cols <= 0 or max_rows <= 0:
        return img_rgb 
        
    total_capacity = max_cols * max_rows
    canvas = np.zeros((max_rows * crop_h, max_cols * crop_w, 3), dtype=np.uint8)
    occupied_mask = np.zeros((H, W), dtype=bool)

    global_selected = []
    max_attempts = total_capacity * 100 
    attempts = 0
    
    while len(global_selected) < total_capacity and attempts < max_attempts:
        cx = np.random.randint(margin_x, W - margin_x)
        cy = np.random.randint(margin_y, H - margin_y)
        
        if not check_overlap_and_mark(cx, cy, crop_w, crop_h, occupied_mask):
            global_selected.append((cx, cy))
            
        attempts += 1

    global_selected.sort(key=lambda pt: (pt[1], pt[0]))
    
    for i, (cx, cy) in enumerate(global_selected):
        row, col = i // max_cols, i % max_cols
        y1, x1 = cy - margin_y, cx - margin_x
        canvas[row*crop_h : (row+1)*crop_h, col*crop_w : (col+1)*crop_w] = \
            img_rgb[y1 : y1 + crop_h, x1 : x1 + crop_w]
            
    return canvas



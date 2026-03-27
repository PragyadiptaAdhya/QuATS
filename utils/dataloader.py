import os
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from PIL import Image
from utils.utils import extract_uniform_random_patches

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def vqa_collate_fn(batch):
    # Bypass standard collation for lists of PIL images
    return batch[0]

def get_video_frames(video_path, num_frames=None):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    orig_fps = vr.get_avg_fps()
    
    if num_frames is None or num_frames >= total_frames:
        indices = np.arange(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    effective_fps = len(indices) / (total_frames / orig_fps)
    processed_images = []
    
    # Extract frame-by-frame to prevent RAM explosion
    for idx in indices:
        frame_array = vr[idx].asnumpy()
        patched_grid = extract_uniform_random_patches(frame_array)
        processed_images.append(Image.fromarray(patched_grid))
        
    return processed_images, effective_fps, total_frames

class VQADataset(Dataset):
    def __init__(self, csv_file, video_dir, config):
        self.config = config
        df = pd.read_csv(csv_file)
        self.samples = [
            (os.path.join(video_dir, f"{row['video_name']}.mp4"),
             1 + 99*((float(row['mos']) - 1)/4.0))
            for _, row in df.iterrows()
        ]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Heavy lifting moved here so PyTorch workers can do this in the background
        video_path, mos = self.samples[idx]
        frames, fps, _ = get_video_frames(video_path, num_frames=self.config.NUM_KEYFRAMES)
        return frames, fps, mos
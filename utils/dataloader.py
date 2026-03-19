import os
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from PIL import Image
from utils.utils import *

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




def get_video_frames(video_path, num_frames=None):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    orig_fps = vr.get_avg_fps()
    
    if num_frames is None or num_frames >= total_frames:
        indices = np.arange(total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    # Calculate effective FPS: (number of extracted frames) / (total original duration)
    # Total original duration = total_frames / orig_fps
    effective_fps = len(indices) / (total_frames / orig_fps)

    # Note: Loading 128 frames of 4K video into RAM at once might cause memory issues.
    # If it crashes, change this to an iterator instead of vr.get_batch()
    frames_batch = vr.get_batch(indices).asnumpy()
    processed_images = []
    
    for frame_array in frames_batch:
        patched_grid = extract_uniform_random_patches(frame_array)
        processed_images.append(Image.fromarray(patched_grid))
        
    return processed_images, effective_fps, total_frames


class VQADataset(Dataset):
    def __init__(self, csv_file, video_dir):
        df = pd.read_csv(csv_file)
        self.samples = [
            (os.path.join(video_dir, f"{row['video_name']}.mp4"),
             1 + 99*((float(row['mos']) - 1)/4.0))
            for _, row in df.iterrows()
        ]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
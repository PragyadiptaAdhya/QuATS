import cv2
import os
from utils.utils import extract_uniform_random_patches

def process_video(video_path, output_dir="outputs_mapped", num_frames=4):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        raise ValueError("Could not determine total frame count.")

    # Select evenly spaced frame indices
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    for idx, frame_idx in enumerate(indices):
        # Jump to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"Skipping frame {frame_idx} (could not read).")
            continue

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run patch extraction
        patched = extract_uniform_random_patches(frame_rgb)

        # Convert back RGB -> BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        patched_bgr = cv2.cvtColor(patched, cv2.COLOR_RGB2BGR)

        # Save
        frame_path = os.path.join(output_dir, f"frame_{idx}.png")
        patched_path = os.path.join(output_dir, f"patched_{idx}.png")

        cv2.imwrite(frame_path, frame_bgr)
        cv2.imwrite(patched_path, patched_bgr)

        print(f"Saved frame {frame_idx} → {frame_path}")
        print(f"Saved patched {frame_idx} → {patched_path}")

    cap.release()


if __name__ == "__main__":
    video_path = ""  
    process_video(video_path)



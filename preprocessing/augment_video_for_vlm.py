import os
import torchvision.transforms as T
import cv2
import numpy as np
from torchvision.transforms import functional as F
import pandas as pd

class VideoAugmenter:
    def __init__(self):
        # Frame-wise augmentations (spatial)
        self.frame_transforms = [
            T.RandomHorizontalFlip(p=1.0),  # Flip left-right
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.1),  # Adjust colors
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))  # Apply blur
        ]

    def augment_frame(self, frame, frame_transform):
        """Applies frame-wise transformations"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB
        frame = F.to_pil_image(frame)  # Convert to PIL for torchvision
        frame = frame_transform(frame)  # Apply spatial transforms
        return np.array(frame)  # Convert back to NumPy array

    def augment_video(self, frames, transform=None):
        """Applies both spatial and temporal augmentations"""
        target_length = 150
        if len(frames) < target_length:
            frames = self.add_frames(frames, target_length=target_length)  # Add frames if needed
        
        # Apply spatial augmentations per frame
        augmented_frames = [self.augment_frame(frame, transform) for frame in frames]
        
        return augmented_frames

    def add_frames(self, frames, target_length=120):
        """Adds frames to videos with less than target_length using interpolation and duplication."""
        num_frames = len(frames)
        if num_frames >= target_length:
            return frames  # No need to add frames

        # Compute how many frames to add
        frames_needed = target_length - num_frames
        
        # Duplicate frames evenly
        new_frames = []
        for i in range(num_frames - 1):
            new_frames.append(frames[i])
            if len(new_frames) < target_length:
                # Interpolate between two frames
                interpolated_frame = (frames[i].astype(np.float32) + frames[i + 1].astype(np.float32)) / 2
                interpolated_frame = interpolated_frame.astype(np.uint8)
                new_frames.append(interpolated_frame)
        
        # Add last frame until we reach the target length
        while len(new_frames) < target_length:
            new_frames.append(frames[-1])

        return new_frames

# Load a sample video
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

# Save augmented video
def save_video(frames, output_path, fps=30):
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

if __name__ == "__main__":
    # Augment on abnormal action videos only in the training set
    base_video_path = "D:/6. Datasets/SPHAR-Dataset/unseen_abnormal_videos"
    df_abnormal_videos = pd.read_csv("../datasets/unseen_abnormal_videos_with_description.csv")

    transform_names = ['hflip', 'jitter', 'blur']

    video_file_names = []
    descriptions = []

    for index, row in df_abnormal_videos.iterrows():
        print(f'Augmenting video {index+1}/{len(df_abnormal_videos)}: {row["video_file_name"]}'.ljust(200), end='\r')
        video_name = row['video_file_name']
        video_path = os.path.join(base_video_path, video_name)
        
        # store original names and descriptions
        video_file_names.append(video_name)
        descriptions.append(row['actual_description'])
        
        # Load video and apply augmentations
        video_frames, fps = load_video(video_path)
        augmenter = VideoAugmenter()

        # Save augmented video
        for i, transform in enumerate(augmenter.frame_transforms):
            augmented_frames = augmenter.augment_video(video_frames, transform=transform)
            saved_video_name = video_name.split(".mp4")[0] + f'_augmented_{transform_names[i]}.mp4'
            output_path = f"{base_video_path}/{saved_video_name}"
            save_video(augmented_frames, output_path, fps)
            # store augmented names and descriptions
            video_file_names.append(saved_video_name)
            descriptions.append(row['actual_description'])
    
    # Save updated dataframe with augmented videos
    df_abnormal_videos = pd.DataFrame({
        'video_file_name': video_file_names,
        'actual_description': descriptions
    })
    df_abnormal_videos.to_csv("../datasets/unseen_abnormal_videos_with_description_augmented.csv", index=False)
    
    print(f"Augmented video completed. Total augmented videos: {len(df_abnormal_videos)}")

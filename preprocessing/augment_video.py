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
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust colors
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))  # Apply blur
        ]

    def augment_frame(self, frame, frame_transform):
        """Applies frame-wise transformations"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB
        frame = F.to_pil_image(frame)  # Convert to PIL for torchvision
        frame = frame_transform(frame)  # Apply spatial transforms
        return np.array(frame)  # Convert back to NumPy array

    def augment_video(self, frames):
        """Applies both spatial and temporal augmentations"""
        target_length = 120
        if len(frames) < target_length:
            frames = self.add_frames(frames, target_length=target_length)  # Add frames if needed
        
        chosen_transform = np.random.choice(self.frame_transforms)
        # Apply spatial augmentations per frame
        augmented_frames = [self.augment_frame(frame, chosen_transform) for frame in frames]
        
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
    base_video_path = "D:/6. Datasets/SPHAR-Dataset/videos"
    df_video_subset_info = pd.read_csv("../datasets/video_subset_information.csv")
    abnormal_train_videos = df_video_subset_info[(df_video_subset_info['label'] == 'abnormal') & (df_video_subset_info['subset'] == 'train')]

    # augement only 50% of abnormal training videos
    abnormal_train_videos_augment = abnormal_train_videos.sample(frac=0.5, random_state=42)

    # Save augmented videos information
    df_augmented_videos = abnormal_train_videos_augment.copy()
    df_augmented_videos['video_file_name'] = df_augmented_videos['video_file_name'].apply(lambda x: x.split(".mp4")[0] + '_augmented.mp4')
    df_augmented_videos.to_csv("../datasets/augmented_abnormal_videos_information.csv", index=False)

    for index, row in abnormal_train_videos_augment.iterrows():
        action = row['action']
        video_name = row['video_file_name']
        video_path = os.path.join(base_video_path, action, video_name)
        
        # Load video and apply augmentations
        video_frames, fps = load_video(video_path)
        augmenter = VideoAugmenter()
        augmented_frames = augmenter.augment_video(video_frames)

        # Save augmented video
        saved_video_name = video_name.split(".mp4")[0] + '_augmented.mp4'
        output_path = f"{base_video_path}/{action}/{saved_video_name}"
        save_video(augmented_frames, output_path, fps)
        print(f"Augmented video saved to {output_path}")

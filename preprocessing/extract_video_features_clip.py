import torch
import cv2
import os
import clip
from PIL import Image
import pandas as pd
import numpy as np
from rembg import remove, new_session
import random

session = new_session('u2net')

def random_background_replace_cv2_batch(frames):
    """
    Replace the background of multiple OpenCV frames with randomly generated ones (batch processing).
    Randomly chooses between:
      1. Random noise background
      2. Random solid color background
    Keeps the foreground (person/object) using rembg batch processing.

    Args:
        frames (list of np.ndarray): List of input frames (RGB format).

    Returns:
        list of np.ndarray: List of output frames (RGB format, with randomized backgrounds).
    """
    if not frames:
        return []
    
    # --- Convert OpenCV frames to PIL Images
    pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)) for frame in frames]
    
    # --- Batch remove backgrounds (RGBA output)
    # rembg's remove function can handle a list of images
    fg_images = [remove(img, session=session).convert("RGBA") for img in pil_images]
    
    # --- Process each foreground with random background
    results = []
    for fg in fg_images:
        w, h = fg.size
        
        # --- Randomly choose background type
        if random.random() < 0.5:
            # Option 1: Random noise background
            bg_array = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        else:
            # Option 2: Random solid color background
            color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            bg_array = np.ones((h, w, 3), dtype=np.uint8) * color

        # --- Convert to PIL RGBA
        bg = Image.fromarray(bg_array).convert("RGBA")

        # --- Composite foreground over background
        result = Image.alpha_composite(bg, fg).convert("RGB")
        results.append(np.array(result))
    
    return results


def random_background_replace_cv2(frame):
    """
    Replace the background of a single OpenCV frame with a randomly generated one.
    This is a wrapper around the batch function for single frame processing.
    
    Args:
        frame (np.ndarray): Input frame (RGB format).

    Returns:
        np.ndarray: Output frame (RGB format, with randomized background).
    """
    results = random_background_replace_cv2_batch([frame])
    return results[0] if results else frame


def extract_video_features(video_path, model, preprocess, device):
    """Extract features from a video using CLIP model."""
    is_replace_bg = False  # Set to True to enable background replacement

    if random.random() > 0.5:
        is_replace_bg = True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}.")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        print(f"No frames extracted from {video_path}.")
        return None
    
    if is_replace_bg:
        print(f"==> Applying random background replacement for video: {video_path}")
        # Use batch processing for better performance
        batch_size_bg = 16  # Process backgrounds in smaller batches
        processed_frames = []
        for i in range(0, len(frames), batch_size_bg):
            batch = frames[i:i+batch_size_bg]
            processed_batch = random_background_replace_cv2_batch(batch)
            processed_frames.extend(processed_batch)
        frames = processed_frames

    # Process frames in smaller batches to avoid GPU memory overflow
    batch_size = 64  # Adjust batch size based on your GPU memory
    video_features = []

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        input_tensors = torch.stack([preprocess(Image.fromarray(frame)).to(device) for frame in batch_frames])
        
        with torch.no_grad():
            batch_features = model.encode_image(input_tensors)
            video_features.append(batch_features.cpu().numpy())

    video_features = np.concatenate(video_features, axis=0)

    return video_features


if __name__ == "__main__":
    df_video_subset = pd.read_csv("../datasets/video_subset_information.csv")
    df_augmented_videos = pd.read_csv("../datasets/augmented_abnormal_videos_information.csv")
    # Combine original subset and augmented videos information
    df_video_info = pd.concat([df_video_subset, df_augmented_videos], ignore_index=True)
    
    base_video_path = "D:/6. Datasets/SPHAR-Dataset/videos"

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    all_video_features = []
    video_names = []
    labels = []
    subsets = []

    for index, row in df_video_info.iterrows():
        try:
            print(f"Processing video ({index+1}/{len(df_video_info)}): {row['video_file_name']} ...")
            action = row['action']
            video_name = row['video_file_name']
            label = row['label']
            subset = row['subset']
            video_path = os.path.join(base_video_path, action, video_name)

            features = extract_video_features(video_path, model, preprocess, device)
            
            if features is None:
                continue

            for feature in features:
                # Make each frame feature a separate entry as column in the dataframe
                feature_dict = {f"feature_{i+1}": feature[i] for i in range(len(feature))}
                all_video_features.append(feature_dict)
            
            video_names.extend([video_name]*len(features))
            labels.extend([label]*len(features))
            subsets.extend([subset]*len(features))
        except Exception as e:
            print(f"Error processing video {row['video_file_name']}: {e}")

    # Save dataframe to csv
    # Since dataframe can be large, save to csv without index and split into multiple files if necessary
    print(f"Saving features to csv files in chunks...")
    chunk_size = 200000  # Adjust based on your memory constraints

    for i in range(0, len(all_video_features), chunk_size):
        df_features_chunk = pd.DataFrame(all_video_features[i:i+chunk_size])
        df_features_chunk['video_file_name'] = video_names[i:i+chunk_size]
        df_features_chunk['label'] = labels[i:i+chunk_size]
        df_features_chunk['subset'] = subsets[i:i+chunk_size]
        df_features_chunk.to_csv(f"../datasets/clip/video_frame_features_clip_part_{i//chunk_size}.csv", index=False)

    print("All video features extracted and saved to ../datasets/clip/video_frame_features_clip.csv in chunks.")
import torch
import cv2
import os
import clip
from PIL import Image
import pandas as pd
import numpy as np

def extract_video_features(video_path, model, preprocess, device):
    """Extract features from a video using CLIP model."""
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

            print(f"[Done] Extracted features from {video_name}, shape: {features.shape}")
        except Exception as e:
            print(f"Error processing video {row['video_file_name']}: {e}")

    # Save dataframe to csv
    # Since dataframe can be large, save to csv without index and split into multiple files if necessary
    print(f"Saving features to csv files in chunks...")
    chunk_size = 300000  # Adjust based on your memory constraints

    for i in range(0, len(all_video_features), chunk_size):
        df_features_chunk = pd.DataFrame(all_video_features[i:i+chunk_size])
        df_features_chunk['video_file_name'] = video_names[i:i+chunk_size]
        df_features_chunk['label'] = labels[i:i+chunk_size]
        df_features_chunk['subset'] = subsets[i:i+chunk_size]
        df_features_chunk.to_csv(f"../datasets/video_frame_features_clip_part_{i//chunk_size}.csv", index=False)

    print("All video features extracted and saved to ../datasets/video_frame_features_clip.csv in chunks.")

import torch
import cv2
import os
import clip
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import timm

def extract_video_features(video_path, model, device):
    """Extract features from a video using ViT model."""

    # Preprocessing for ViT (matches ImageNet training)
    vit_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
    ])

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
    batch_size = 32  # Adjust batch size based on your GPU memory
    video_features = []

    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]

        with torch.no_grad():
            processed = torch.stack([vit_transform(Image.fromarray(f)) for f in batch_frames])
            processed = processed.to(device)
            feats = model(processed)
            video_features.append(feats.cpu().numpy())

    video_features = np.concatenate(video_features, axis=0)

    return video_features


if __name__ == "__main__":
    ''' Extract and save video frame features for all videos in the dataset.'''
    ''' This use ViT pretrained model to extract features from each frame of the video.'''

    df_video_subset = pd.read_csv("../datasets/video_subset_information.csv")
    df_augmented_videos = pd.read_csv("../datasets/augmented_abnormal_videos_information.csv")
    # Combine original subset and augmented videos information
    df_video_info = pd.concat([df_video_subset, df_augmented_videos], ignore_index=True)
    
    base_video_path = "D:/6. Datasets/SPHAR-Dataset/videos"

    # Load ViT pretrained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load pretrained ViT (e.g., ViT-B/16 from ImageNet-21k)
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)

    # Remove classification head -> we only want embeddings
    vit_model.reset_classifier(0)  # removes the final FC layer
    
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

            features = extract_video_features(video_path, vit_model, device)
            
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
    chunk_size = 100000  # Adjust based on your memory constraints

    for i in range(0, len(all_video_features), chunk_size):
        print(f"Saving chunk {i//chunk_size} ...")
        df_features_chunk = pd.DataFrame(all_video_features[i:i+chunk_size])
        df_features_chunk['video_file_name'] = video_names[i:i+chunk_size]
        df_features_chunk['label'] = labels[i:i+chunk_size]
        df_features_chunk['subset'] = subsets[i:i+chunk_size]
        df_features_chunk.to_csv(f"../datasets/vit/video_frame_features_vit_part_{i//chunk_size}.csv", index=False)

    print("All video features extracted and saved to ../datasets/vit/video_frame_features_vit.csv")

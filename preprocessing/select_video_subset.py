import os
import numpy as np
import pandas as pd

events = {
    "normal": ["sitting", "walking", "running", "neutral", "luggage"],
    "abnormal": ["carcrash", "hitting", "kicking", "falling", "vandalizing", "panicking", "stealing", "murdering", "igniting"]
}

def select_video_subset(base_path):
    random_seed = 42
    video_file_names, video_actions, video_labels = [], [], []

    for event in events.keys():
        actions = events[event]
        for action in actions:
            print(f"Processing action: {action}")
            videos = os.listdir(os.path.join(base_path, action))

            # Select 70% of videos randomly    
            num_selected = int(len(videos) * 0.7)
            np.random.seed(random_seed)
            selected_videos = np.random.choice(videos, size=num_selected, replace=False)

            for selected_video in selected_videos:
                video_file_names.append(selected_video)
                video_actions.append(action)
                video_labels.append(event)
        
    # Save into csv file
    df_video_subset = pd.DataFrame({
        "video_file_name": video_file_names,
        "action": video_actions,
        "label": video_labels
    })

    # Balance the dataset by ensuring equal number of normal and abnormal videos
    n_abnormal = len(df_video_subset[df_video_subset['label'] == 'abnormal'])

    n_sample_normal = int(n_abnormal + (n_abnormal * 0.6 * 0.5)) # add 50% of augmented abnormal videos that is subset of train set
    df_video_subset = df_video_subset.groupby('label').apply(lambda x: x.sample(n=n_sample_normal if x.name == "normal" else n_abnormal, random_state=random_seed)).reset_index(drop=True)

    # Create train, val, test splits for both normal and abnormal classes
    subset = []

    # Since abnormal videos are less, we use 60% for training, 20% for validation, and 20% for testing before augmentation
    # After augmentation, the training set will have 50% more on abnormal videos
    for label in df_video_subset['label'].unique():
        n_videos = len(df_video_subset[df_video_subset['label'] == label])
        
        if label == "abnormal":
            train_ratio, val_ratio = 0.6, 0.2
        else:
            train_ratio, val_ratio = 0.7, 0.15
        
        n_train = int(n_videos * train_ratio)
        n_val = int(n_videos * val_ratio)
        n_test = n_videos - n_train - n_val
        
        subset += ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
    
    df_video_subset['subset'] = subset
    df_video_subset.to_csv(f"../datasets/video_subset_information.csv", index=False)


if __name__ == "__main__":
    '''Select a subset of videos from each action category.'''
    base_path = "D:/6. Datasets/SPHAR-Dataset/videos"
    select_video_subset(base_path)

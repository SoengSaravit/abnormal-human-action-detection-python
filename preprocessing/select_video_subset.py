import os
import numpy as np
import pandas as pd

events = {
    "normal": ["sitting", "walking", "running", "neutral", "luggage"],
    "abnormal": ["hitting", "kicking", "falling", "vandalizing", "panicking", "stealing", "murdering", "igniting"] # exclude carcrash action due not highly related to human action
}

def select_video_subset(base_path):
    random_seed = 42
    video_file_names, video_actions, video_labels = [], [], []

    for event in events.keys():
        actions = events[event]
        for action in actions:
            print(f"Processing action: {action}")
            videos = os.listdir(os.path.join(base_path, action))

            # Select videos randomly
            ratio = 0.80 if event == "abnormal" else 1.0    
            num_selected = int(len(videos) * ratio)
            np.random.seed(random_seed)
            selected_videos = np.random.choice(videos, size=num_selected, replace=False)

            for selected_video in selected_videos:
                skip_dataset = ["ucaerial", "aerial", "rooftop"] # Skip these datasets or any video or with aerial or rooftop view due to video quality issue
                is_skip = False
                for name in skip_dataset:
                    if name in selected_video:
                        is_skip = True
                        break
                if is_skip:
                    continue
                video_file_names.append(selected_video)
                video_actions.append(action)
                video_labels.append(event)
        
    # Save into csv file
    df_video_subset = pd.DataFrame({
        "video_file_name": video_file_names,
        "action": video_actions,
        "label": video_labels
    })

    # Shuffle the dataframe
    df_video_subset = df_video_subset.sample(frac=1, random_state=random_seed)

    # Create train, val, test splits for both normal and abnormal classes
    subset = []
    df_tmps = []

    for label in df_video_subset['label'].unique():
        df_tmp = df_video_subset[df_video_subset['label'] == label]
        df_tmps.append(df_tmp)
        n_videos = len(df_tmp)

        if label == 'abnormal':
            train_ratio, val_ratio = 0.7, 0.15
        else:
            train_ratio, val_ratio = 0.6, 0.2

        n_train = int(n_videos * train_ratio)
        n_val = int(n_videos * val_ratio)
        n_test = n_videos - n_train - n_val
        
        subset = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
        df_tmp['subset'] = subset

    df_video_subset_new = pd.concat(df_tmps, axis=0)
    df_video_subset_new.to_csv(f"../datasets/video_subset_information.csv", index=False)


if __name__ == "__main__":
    '''Select a subset of videos from each action category.'''
    base_path = "D:/6. Datasets/SPHAR-Dataset/videos"
    select_video_subset(base_path)

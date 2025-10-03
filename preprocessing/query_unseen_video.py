import pandas as pd
import os


events = {
    "normal": ["sitting", "walking", "running", "neutral", "luggage"],
    "abnormal": ["carcrash", "hitting", "kicking", "falling", "vandalizing", "panicking", "stealing", "murdering", "igniting"]
}

if __name__ == '__main__':
    ''' 
        This script is used to query unseen videos that are not included in training from the dataset directory
        and save the information to a csv file.
    '''
    
    base_path = "D:/6. Datasets/SPHAR-Dataset/videos"
    
    df_video_subset = pd.concat([pd.read_csv("../datasets/video_subset_information.csv"), pd.read_csv("../datasets/augmented_abnormal_videos_information.csv")], ignore_index=True)
    
    video_file_names = df_video_subset['video_file_name'].tolist()
    unseen_videos, video_type, labels = [], [], []

    for event in events.keys():
        actions = events[event]
        for action in actions:
            print(f"Processing action: {action}")
            videos = os.listdir(os.path.join(base_path, action))

            for video in videos:
                if video not in video_file_names:
                    unseen_videos.append(video)
                    video_type.append(action)
                    labels.append(event)

    df_unseen_videos = pd.DataFrame({
        "video_file_name": unseen_videos,
        "video_type": video_type,
        "label": labels
    })

    df_unseen_videos.to_csv(f"../datasets/unseen_videos_information.csv", index=False)
    print(f"Total unseen videos: {len(df_unseen_videos)}")

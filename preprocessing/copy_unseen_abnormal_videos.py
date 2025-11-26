# Script for copying unseen abnormal videos to a designated folder
import os
import shutil
import pandas as pd

# Function to copy unseen abnormal video to target directory
def copy_unseen_abnormal_videos(source_path, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # copy file under source_path to target_dir
    shutil.copy(source_path, target_dir)

if __name__ == "__main__":
    # Read the CSV file containing unseen abnormal video paths
    df_unseen = pd.read_csv('../datasets/unseen_videos_information.csv')
    df_unseen_abnormal = df_unseen[df_unseen['label'] == 'abnormal'].reset_index(drop=True)

    base_path = 'D:/6. Datasets/SPHAR-Dataset/videos'
    target_directory = 'D:/6. Datasets/SPHAR-Dataset/unseen_abnormal_videos/'

    for index, row in df_unseen_abnormal.iterrows():
        video_path = os.path.join(base_path, row['video_type'] , row['video_file_name'])
        copy_unseen_abnormal_videos(video_path, target_directory)
        print(f"{index + 1}/{len(df_unseen_abnormal)}: Copied {video_path} to {target_directory}")
        
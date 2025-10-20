import sys
sys.path.append('./notebooks')

import os
import pandas as pd
from utils.abnormal_action_detector import AbnormalActionDetector

if __name__ == '__main__':
    base_path = "D:/6. Datasets/SPHAR-Dataset-1.0/videos" # Original dataset without extra trimming or augmented videos
    df_unseen_videos = pd.read_csv("datasets/unseen_videos_information.csv")
    df_abnormal_videos = df_unseen_videos[df_unseen_videos['label'] == 'abnormal']
    df_normal_videos = df_unseen_videos[df_unseen_videos['label'] == 'normal']

    # video = df_normal_videos.iloc[125]
    video = df_abnormal_videos.iloc[200]
    video_name = video['video_file_name'].replace(" - Trim", "").replace("_trimmed", "")
    video_source = os.path.join(base_path, video['video_type'], video_name)
    video_source = "D:/6. Datasets/Fall-Dataset/Cut/C00_010_0004-C1.mp4"
    # video_source = "D:/6. Datasets/Fall-Dataset/Cut/C00_051_0002-C1.mp4"
    # video_source = "D:/6. Datasets/Fall-Dataset/Cut/C00_001_0018-C1.mp4"
    # video_source = "D:/6. Datasets/Fall-Dataset/Cut/C00_096_0008-C1.mp4"
    # video_source = "D:/6. Datasets/Fall-Dataset/video-fall-1.mp4"
    # video_source = "D:/6. Datasets/Fall-Dataset/video-fall-2.mp4"
    # video_source = "D:/6. Datasets/Fall-Dataset/video-fall-3.mp4"
    # video_source = "D:/6. Datasets/Fall-Dataset/video-fall-5.mp4"
    # video_source = "D:/6. Datasets/Fight-Dataset/fight-video-2.mp4"
    # video_source = "D:/6. Datasets/Fight-Dataset/C00_083_0003-C1.mp4"

    print(f"Processing video: {video_source}")
    image_encoder_type = "clip"  # "clip" or "vit"
    model, model_version = ("transformer", "v1")
    detector = AbnormalActionDetector(f"models/{model}_model_{model_version}.pt", 
                                    window_size=120,
                                    lag_sampling=5, # Lag Sampling Stride (which samples within a window)
                                    abnormal_threshold=0.05,
                                    image_encoder_type=image_encoder_type)
    
    detector.detect_abnormal_action(video_source, is_save_result=False)
    # print(detector.get_abnormal_action_detection_results(source=video_source))

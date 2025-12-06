import sys
sys.path.append('./notebooks')

from utils.abnormal_action_detector import AbnormalActionDetector
import pandas as pd
from sklearn.metrics import accuracy_score
import os
import numpy as np

if __name__ == '__main__':
    image_encoder_type = "clip"  # "clip" or "vit"
    model, model_version = ("transformer", "v3")
    detector = AbnormalActionDetector(f"models/{model}_model_{model_version}.pt", 
                                    window_size=120,
                                    lag_sampling=5, # Lag Sampling Stride (which samples within a window)
                                    abnormal_threshold=0.05,
                                    image_encoder_type=image_encoder_type)
    
    # read unseen abnormal abnormal video
    base_path = "D:\\6. Datasets\\SPHAR-Dataset\\unseen_abnormal_videos"
    df_unseen_videos = pd.read_csv("datasets/unseen_videos_information.csv")
    df_abnormal_videos = df_unseen_videos[df_unseen_videos['label'] == 'abnormal']
    
    videos = []
    actions = []
    class_indexes = []
    action_classes = []
    confidences = []
    ground_truths = []
    conditions = []

    for i, video in df_abnormal_videos.iterrows():
        video_name = video['video_file_name']
        video_source = os.path.join(base_path, video_name)
        
        class_idx, action_class, conf= detector.get_abnormal_action_detection_results(video_source)
        
        if class_idx is not None:
            videos.append(video_name)
            actions.append(video['video_type'])
            class_indexes.append(class_idx)
            action_classes.append(action_class)
            confidences.append(conf)
            ground_truths.append(1)  # abnormal class index is 1
            conditions.append("Normal") # video condition is normal
            print(f"==> Class: {action_class}, Confidence: {conf}")
    
    # process augmented videos
    augment_type = ["low_illumination", "low_resolution", "partial_occlusion"]
    video_path = "D:/6. Datasets/SPHAR-Dataset/augmented_videos_experiments"
    for video_name in os.listdir(video_path):
        video_source = os.path.join(video_path, video_name)
        
        class_idx, action_class, conf= detector.get_abnormal_action_detection_results(video_source)
        
        if class_idx is not None:
            videos.append(video_name)
            actions.append("Not Specified")
            class_indexes.append(class_idx)
            action_classes.append(action_class)
            confidences.append(conf)
            ground_truths.append(1)
            for aug in augment_type:
                if aug in video_name:
                    conditions.append(aug.replace("_", " ").title())
                    break
            print(f"==> Class: {action_class}, Confidence: {conf}")

    df_results = pd.DataFrame({
        'video_name': videos,
        'action': actions,
        'ground_truth': ground_truths,
        'class_index': class_indexes,
        'action_class': action_classes,
        'avg_confidence': confidences,
        'video_condition': conditions,
    })

    print(f"Overall Accuracy: {accuracy_score(ground_truths, class_indexes)}")
    df_results.to_csv(f'outputs/unseen_abnormal_detection_{model}_{model_version}_results_v2.csv', index=False)
    
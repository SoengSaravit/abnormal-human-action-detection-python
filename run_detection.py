import sys
sys.path.append('./notebooks')

import os
from utils.abnormal_action_detector import AbnormalActionDetector

if __name__ == '__main__':
    base_path = "D:/6. Datasets/SPHAR-Dataset/videos/stealing"
    videos = os.listdir(base_path)
    # video_source = os.path.join(base_path, videos[100])
    video_source = "D:/6. Datasets/Fall-Dataset/Cut/C00_010_0004-C1.mp4"
    # video_source = "D:/6. Datasets/Fall-Dataset/Cut/C00_051_0002-C1.mp4"
    # video_source = "D:/6. Datasets/Fight-Dataset/fight-video-2.mp4"
    # video_source = "D:/6. Datasets/Fight-Dataset/C00_083_0003-C1.mp4"

    image_encoder_type = "clip"  # "clip" or "vit"
    model, model_version = ("transformer", "v3")
    detector = AbnormalActionDetector(f"models/{model}_model_{model_version}.pt", 
                                    window_size=120,
                                    lag_sampling=4, # Lag Sampling Stride (which samples within a window)
                                    abnormal_threshold=0.25,
                                    image_encoder_type=image_encoder_type)
    
    detector.detect_abnormal_action(video_source, is_save_result=False)

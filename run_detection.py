import sys
sys.path.append('./notebooks')

import os
from utils.abnormal_action_detector import AbnormalActionDetector

if __name__ == '__main__':
    base_path = "D:/6. Datasets/SPHAR-Dataset/videos/sitting"
    videos = os.listdir(base_path)
    video_source = os.path.join(base_path, videos[30])
    
    model, model_version = ("transformer", "v1")
    detector = AbnormalActionDetector(f"models/{model}_model_{model_version}.pt", 
                                    window_size=30,
                                    abnormal_threshold=0.25)
    
    detector.detect_abnormal_action(video_source, is_save_result=False)

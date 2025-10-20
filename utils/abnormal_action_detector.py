import sys
sys.path.append('..')

import cv2
from collections import deque
import numpy as np
import pandas as pd
import torch
import datetime
import time
import clip
from PIL import Image
import timm
import torchvision.transforms as transforms


class AbnormalActionDetector():
    def __init__(self, model_path, window_size=30, lag_sampling=1, abnormal_threshold=0.25, image_encoder_type='clip'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_encoder_type = image_encoder_type
        if self.image_encoder_type == 'clip':
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        elif self.image_encoder_type == 'vit':
            self.model = timm.create_model("vit_base_patch16_224", pretrained=True).to(self.device)
            self.model.reset_classifier(0)
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                    std=(0.5, 0.5, 0.5))
            ])
        self.cls_model = torch.load(model_path, weights_only=False)
        self.window_size = window_size
        self.lag_sampling = lag_sampling
        self.effective_window_size = int(window_size / lag_sampling)
        self.frame_histories = deque(maxlen=self.window_size)
        self.classes = ["normal", "abnormal"]  # 0: normal, 1: abnormal
        self.abnormal_threshold = abnormal_threshold
    
    # function fore detecting abnormal gait and display the result in real-time
    def detect_abnormal_action(self, source, is_save_result=False):
        pred_classes = np.array([])
        pred_confs = {0: [], 1: []}
        
        cap = cv2.VideoCapture(source)
        
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        current_time = 0

        st = time.time()
        frame_count = 0

        if is_save_result:
            video_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            print(f"Saving the result to outputs/{video_name}.mp4")
            out = cv2.VideoWriter(f'outputs/{video_name}.mp4', fourcc, video_fps, (video_width, video_height))    

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = self.preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image) if self.image_encoder_type == 'clip' else self.model(image)
            self.frame_histories.append(image_features)
                
            if len(self.frame_histories) == self.window_size:
                with torch.no_grad():
                    indices = np.linspace(0, self.window_size-1, self.effective_window_size, dtype=int)
                    input_tensors = [list(self.frame_histories)[i] for i in indices]
                    X = torch.stack(input_tensors, dim=0)
                    X = X.float().view(1, self.effective_window_size, 512 if self.image_encoder_type == 'clip' else 768).to(self.device)
                    outputs = self.cls_model(X)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    conf_prob = probs[0] if probs[0] > 0.5 else 1 - probs[0]
                    y_pred = probs[0].round()
                    
                    # check if the confidence and prediction is not NaN
                    if not np.isnan([conf_prob, y_pred]).all():
                        # Append the confidence and class to the respective class
                        y_pred = int(y_pred)
                        pred_classes = np.append(pred_classes, y_pred)
                        pred_confs[y_pred].append(conf_prob)
                    
                    # Determine the most frequent predicted class so far
                    # check if there are any predictions
                    if len(pred_classes) > 0:
                        unique, counts = np.unique(pred_classes, return_counts=True)
                        # check if counts of abnormal class is greater than the threshold
                        if len(unique) == 1:
                            majority_class_index = int(unique[0]) # this means there is only one class detected
                        else:
                            majority_class_index = 1 if counts[1]/np.sum(counts) > self.abnormal_threshold else 0
                        majority_class = self.classes[majority_class_index]
                        confidence = np.mean(pred_confs[majority_class_index]) * 100
                        
                        alert_color = (0, 0, 255) if majority_class_index == 1 else (0, 150, 0) # Red for abnormal, Green for normal
                        
            frame_count += 1

            # Calculate average FPS
            current_time = time.time()
            if frame_count > 0 and (current_time - st) > 0:
                avg_fps = frame_count / (current_time - st)
            else:
                avg_fps = avg_fps if 'avg_fps' in locals() else 0
            # Draw the prediction text
            if len(self.frame_histories) >= self.window_size:
                cv2.rectangle(frame, (0, 0), (450 if majority_class_index == 1 else 430, 30), alert_color, -1)
                cv2.putText(frame, f"Prediction: {majority_class}, Confidence: {confidence:.2f}%, FPS: {avg_fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if is_save_result:
                out.write(frame)
            
            cv2.imshow("Abnormal Action Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if is_save_result:
            out.release()
        cap.release()
        cv2.destroyAllWindows()
        
        # Print average FPS
        print(f"Average FPS: {avg_fps:.2f}")

        print(f"Final prediction: {majority_class}, confidence: {confidence:.2f}%")

        return None
    

    # function for detecting abnormal action by producing prediction class and confidence
    def get_abnormal_action_detection_results(self, source):
        print(f"Processing video: {source} ...")
        pred_classes = np.array([])
        pred_confs = {0: [], 1: []}

        # reset frame histories
        self.frame_histories = deque(maxlen=self.window_size)
        
        cap = cv2.VideoCapture(source)
        majority_class_index = None
        majority_class = None
        confidence = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = self.preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image) if self.image_encoder_type == 'clip' else self.model(image)
            self.frame_histories.append(image_features)
                
            if len(self.frame_histories) == self.window_size:
                with torch.no_grad():
                    indices = np.linspace(0, self.window_size-1, self.effective_window_size, dtype=int)
                    input_tensors = [list(self.frame_histories)[i] for i in indices]
                    X = torch.stack(input_tensors, dim=0)
                    X = X.float().view(1, self.effective_window_size, 512 if self.image_encoder_type == 'clip' else 768).to(self.device)
                    outputs = self.cls_model(X)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    conf_prob = probs[0] if probs[0] > 0.5 else 1 - probs[0]
                    y_pred = probs[0].round()
                    
                    # check if the confidence and prediction is not NaN
                    if not np.isnan([conf_prob, y_pred]).all():
                        # Append the confidence and class to the respective class
                        y_pred = int(y_pred)
                        pred_classes = np.append(pred_classes, y_pred)
                        pred_confs[y_pred].append(conf_prob)
                      
        cap.release()
        # Determine the most frequent predicted class so far
        # check if there are any predictions
        if len(pred_classes) > 0:
            unique, counts = np.unique(pred_classes, return_counts=True)
        # check if counts of abnormal class is greater than the threshold
            if len(unique) == 1:
                majority_class_index = int(unique[0]) # this means there is only one class detected
            else:
                majority_class_index = 1 if counts[1]/np.sum(counts) > self.abnormal_threshold else 0
            majority_class = self.classes[majority_class_index]
            confidence = np.mean(pred_confs[majority_class_index]) * 100

        return majority_class_index, majority_class, confidence
    
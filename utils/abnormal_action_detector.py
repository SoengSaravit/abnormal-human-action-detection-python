import sys
from xml.parsers.expat import model
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


class AbnormalActionDetector():
    def __init__(self, model_path, window_size=30, abnormal_threshold=0.25):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.cls_model = torch.load(model_path)
        self.frame_histories = deque(maxlen=window_size)
        self.window_size = window_size
        self.classes = ["abnormal", "normal"]
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

        previous_time = 0
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
            
            # Skip frame to improve performance to get only 10 fps from 30 fps video
            if frame_count % 3 == 0:
                self.frame_histories.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if len(self.frame_histories) == self.window_size:
                    input_tensors = torch.stack([self.clip_preprocess(Image.fromarray(frame)).to(self.device) for frame in self.frame_histories])

                    with torch.no_grad():
                        batch_features = self.clip_model.encode_image(input_tensors)
                        X = batch_features.clone().detach().float().view(1, self.window_size, 512).to(self.device)
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
                                majority_class_index = 0 if counts[0]/np.sum(counts) > self.abnormal_threshold else 1
                            majority_class = self.classes[majority_class_index]
                            confidence = np.mean(pred_confs[majority_class_index]) * 100
                            
                            alert_color = (0, 0, 255) if majority_class_index == 0 else (0, 150, 0) # Red for abnormal, Green for normal
                        
            frame_count += 1
            # Draw the prediction text
            if len(self.frame_histories) >= self.window_size:
                cv2.rectangle(frame, (0, 0), (350, 30), alert_color, -1)
                cv2.putText(frame, f"Prediction: {majority_class}, Confidence: {confidence:.2f}%", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calculate real-time FPS
            current_time = time.time()
            # time_diff = current_time - previous_time
            # if time_diff > 0:
            #     fps = 1 / time_diff
            # previous_time = current_time
            # Calculate average FPS
            if frame_count > 0 and (current_time - st) > 0:
                avg_fps = frame_count / (current_time - st)
            else:
                avg_fps = 0

            cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if is_save_result:
                out.write(frame)
            
            cv2.imshow("Abnormal Gait Detection", frame)
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
    

    # function for detecting abnormal gait by producing prediction class and confidence
    def get_abnormal_action_detection_results(self, source, classes=[0], conf=0.5):
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
            results = self.yolo_model.predict(frame, classes=classes, conf=conf)
            keypoints_xyn = results[0].keypoints.xyn.cpu().numpy()
            # If no keypoints are detected or image with multiple person, skip the image
            if keypoints_xyn.size == 0 or len(keypoints_xyn) > 1:
                continue

            data = self._extract_pose_features(keypoints_xyn[0])
            self.frame_histories.append(data)
            if len(self.frame_histories) == self.window_size:
                df = pd.DataFrame(self.frame_histories)
                df.fillna(method='ffill', inplace=True)
                frame_series = np.array(df).flatten()
                frame_series_scaled = self.scaler.transform(frame_series.reshape(1, -1))
                X = torch.tensor(frame_series_scaled).float().view(1, self.window_size, 9).to(torch.device('cuda'))
                with torch.no_grad():
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
        # check if there are any predictions
        if len(pred_classes) > 0:
            unique, counts = np.unique(pred_classes, return_counts=True)
            # check if counts of abnormal class is greater than the threshold
            if len(unique) == 1:
                majority_class_index = int(unique[0]) # this means there is only one class detected
            else:
                majority_class_index = 0 if counts[0]/np.sum(counts) > self.abnormal_threshold else 1
            majority_class = self.classes[majority_class_index]
            confidence = np.mean(pred_confs[majority_class_index])

        return majority_class_index, majority_class, confidence
    
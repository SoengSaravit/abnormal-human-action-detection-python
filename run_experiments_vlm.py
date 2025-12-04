import torch
import numpy as np
from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration, AutoModelForImageTextToText
import cv2
import pandas as pd
import os
from utils.vlm_evaluation_metrics import compute_bleu_cider_meteor_single_ref
import time
from peft import PeftModel

# Function to read video frames using OpenCV
def read_video_opencv(video_path, num_frames=8):
    '''
    Decode video frames using OpenCV.
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to decode.
    Returns:
        result (np.ndarray): Array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    index = 0
    video = cv2.VideoCapture(video_path)
    total_num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.arange(0, total_num_frames, total_num_frames / num_frames).astype(int)
    while video.isOpened():
        success, frame = video.read()
        if index in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        if success:
            index += 1
        if index >= total_num_frames:
            break

    video.release()

    if not frames:
        raise ValueError("No frames were read from the video.")

    return np.stack(frames)

# Function to run experiments with LLaVA-NeXT-Video model
def run_experiments_llava(df_abnormal_videos):
    # Define model ID
    # model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    model_id = "saravit-soeng/llava-next-video-7b-hf-abnormal-action-fine-tuned-v4" # fine-tuned model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the processor and model
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        low_cpu_mem_usage=True,
        load_in_4bit=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", "text": "What is abnormal action in this video?",
                },
                {
                    "type": "video"
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    base_path = "D:/6. Datasets/SPHAR-Dataset/unseen_abnormal_videos"
    video_names = []
    video_types = []
    video_actual_descriptions = []
    predicted_descriptions = []
    average_inference_time_per_video = []

    try:
        for index, row in  df_abnormal_videos.iterrows():
            st_time = time.time()
            video_name = row['video_file_name']
            video_type = row['video_type']
            video_actual_description = row['actual_description']
            video_path = os.path.join(base_path, video_name)
            print(f"Processing video {index + 1}/{len(df_abnormal_videos)}: {video_name} of type: {video_type}")

            clip = read_video_opencv(video_path, num_frames=8)

            inputs = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(device)

            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            predicted_description = processor.decode(outputs[0], skip_special_tokens=True)
            predicted_description = predicted_description.split("ASSISTANT:")[-1].strip()
            predicted_description = ".".join(predicted_description.split(".")[:-1]).strip() + "."
            end_time = time.time()
            avg_inference_time = round(end_time - st_time)
            
            average_inference_time_per_video.append(avg_inference_time)
            video_names.append(video_name)
            video_types.append(video_type)
            video_actual_descriptions.append(video_actual_description)
            predicted_descriptions.append(predicted_description)
    except Exception as e:
        print(f"Error processing video {video_name}: {e}")

    metrics = compute_bleu_cider_meteor_single_ref(predicted_descriptions, video_actual_descriptions)
    print("Evaluation Metrics:")
    print(f"BLEU-4 Mean: {metrics['BLEU_4_mean']}")
    print(f"CIDEr Mean: {metrics['CIDEr_mean']}")
    print(f"METEOR Mean: {metrics['METEOR_mean']}")

    df_results = pd.DataFrame({
        'video_file_name': video_names,
        'video_type': video_types,
        'actual_description': video_actual_descriptions,
        'predicted_description': predicted_descriptions,
        'BLEU_4': metrics['BLEU_per_sample'],
        'CIDEr': metrics['CIDEr_per_sample'],
        'METEOR': metrics['METEOR_per_sample'],
        'average_inference_time': average_inference_time_per_video
    })

    df_results.to_csv('outputs/llava_next_video_fine_tuned_7b_abnormal_videos_experiment_results_v4.csv', index=False)
    print("Inference results saved to CSV.")


# Function to run experiments with SmolVLM2 model
def run_experiments_smolvlm2(df_abnormal_videos):
    # Define model ID
    model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the processor and model
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        low_cpu_mem_usage=True,
        load_in_4bit=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)

    base_path = "D:/6. Datasets/SPHAR-Dataset/unseen_abnormal_videos"
    video_names = []
    video_types = []
    video_actual_descriptions = []
    predicted_descriptions = []
    average_inference_time_per_video = []

    try:
        for index, row in  df_abnormal_videos.iterrows():
            st_time = time.time()
            video_name = row['video_file_name']
            video_type = row['video_type']
            video_actual_description = row['actual_description']
            video_path = os.path.join(base_path, video_name)
            print(f"Processing video {index + 1}/{len(df_abnormal_videos)}: {video_name} of type: {video_type}")

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", "text": "What is abnormal action in this video?",
                        },
                        {
                            "type": "video", "path": video_path
                        }
                    ]
                }
            ]

            inputs = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)

            outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            predicted_description = processor.decode(outputs[0], skip_special_tokens=True)
            predicted_description = predicted_description.split("Assistant:")[-1].strip()
            predicted_description = ".".join(predicted_description.split(".")[:-1]).strip() + "."
            end_time = time.time()
            avg_inference_time = round(end_time - st_time)
            
            average_inference_time_per_video.append(avg_inference_time)
            video_names.append(video_name)
            video_types.append(video_type)
            video_actual_descriptions.append(video_actual_description)
            predicted_descriptions.append(predicted_description)
    except Exception as e:
        print(f"Error processing video {video_name}: {e}")

    metrics = compute_bleu_cider_meteor_single_ref(predicted_descriptions, video_actual_descriptions)
    print("Evaluation Metrics:")
    print(f"BLEU-4 Mean: {metrics['BLEU_4_mean']}")
    print(f"CIDEr Mean: {metrics['CIDEr_mean']}")
    print(f"METEOR Mean: {metrics['METEOR_mean']}")

    df_results = pd.DataFrame({
        'video_file_name': video_names,
        'video_type': video_types,
        'actual_description': video_actual_descriptions,
        'predicted_description': predicted_descriptions,
        'BLEU_4': metrics['BLEU_per_sample'],
        'CIDEr': metrics['CIDEr_per_sample'],
        'METEOR': metrics['METEOR_per_sample'],
        'average_inference_time': average_inference_time_per_video
    })

    df_results.to_csv('outputs/smolvlm2_2b_abnormal_videos_experiment_results.csv', index=False)
    print("Inference results saved to CSV.")
    

if __name__ == '__main__':
    # read unseen abnormal video
    df_unseen_abnormal_videos = pd.read_csv("datasets/unseen_abnormal_videos_with_description.csv")

    run_experiments_llava(df_unseen_abnormal_videos)
    # run_experiments_smolvlm2(df_unseen_abnormal_videos)

import cv2
import os

# Function to extract frames from a video
def extract_frames(video_path, output_folder, file_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_filename = os.path.join(output_folder, f"{file_name}_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)

    cap.release()
    print(f"[Video: {file_name}] Extracted {frame_count} frames to {output_folder}")

if __name__ == '__main__':
    '''
    This script extracts frames from the videos and saves them as images.
    '''
    ds_base_path = 'D:/6. Datasets/SPHAR-Dataset'
    folders = os.listdir(f'{ds_base_path}/videos')
    for folder in folders:
        videos = os.listdir(f'{ds_base_path}/videos/{folder}')
        for video in videos:
            file_name = video.split('.mp4')[0]
            video_path = f'{ds_base_path}/videos/{folder}/{video}'
            output_folder = f'{ds_base_path}/images/{folder}'
            extract_frames(video_path, output_folder, file_name)
    print("All frames extracted!")

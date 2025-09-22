import cv2
import os

def trim_video(input_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Skip videos with 150 frames or less
    if total_frames <= 150:
        print("Video has 150 frames or less; skipping trimming.")
        cap.release()
        return

    # Determine trimming rates based on total frames
    # 300 frames correspond to 30 seconds at 30 fps
    if total_frames > 300:
        start_trim_rate = 0.4
        end_trim_rate = 0.9
    else:
        start_trim_rate = 0.2
        end_trim_rate = 0.8


    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate start and end frames
    start_frame = int(total_frames * start_trim_rate)  
    end_frame = int(total_frames * end_trim_rate)

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read and write frames within the trimmed range
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Trimmed video saved to {output_path}")

if __name__ == "__main__":
    subset_path = "D:/6. Datasets/SPHAR-Dataset/videos/vandalizing"
    videos = os.listdir(subset_path)

    for i, video in enumerate(videos):        
        input_video_path = f"{subset_path}/{video}"
        output_video_path = f"{input_video_path.split('.mp4')[0]}_trimmed.mp4"
        trim_video(input_video_path, output_video_path)

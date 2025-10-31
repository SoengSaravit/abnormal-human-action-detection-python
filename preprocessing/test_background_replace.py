import cv2
import os
import sys
import numpy as np
from PIL import Image
from rembg import remove, new_session
import random

session = new_session('u2net')

def random_background_replace_cv2_batch(frames):
    """
    Replace the background of multiple OpenCV frames with randomly generated ones (batch processing).
    Randomly chooses between:
      1. Random noise background
      2. Random solid color background
    Keeps the foreground (person/object) using rembg batch processing.

    Args:
        frames (list of np.ndarray): List of input frames (RGB format).

    Returns:
        list of np.ndarray: List of output frames (RGB format, with randomized backgrounds).
    """
    if not frames:
        return []
    
    # --- Convert OpenCV frames to PIL Images
    pil_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)) for frame in frames]
    
    # --- Batch remove backgrounds (RGBA output)
    # rembg's remove function can handle a list of images
    fg_images = [remove(img, session=session).convert("RGBA") for img in pil_images]
    
    # --- Process each foreground with random background
    results = []
    for fg in fg_images:
        w, h = fg.size
        
        # --- Randomly choose background type
        if random.random() < 0.5:
            # Option 1: Random noise background
            bg_array = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        else:
            # Option 2: Random solid color background
            color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            bg_array = np.ones((h, w, 3), dtype=np.uint8) * color

        # --- Convert to PIL RGBA
        bg = Image.fromarray(bg_array).convert("RGBA")

        # --- Composite foreground over background
        result = Image.alpha_composite(bg, fg).convert("RGB")
        results.append(np.array(result))
    
    return results


def random_background_replace_cv2(frame):
    """
    Replace the background of a single OpenCV frame with a randomly generated one.
    This is a wrapper around the batch function for single frame processing.
    
    Args:
        frame (np.ndarray): Input frame (RGB format).

    Returns:
        np.ndarray: Output frame (RGB format, with randomized background).
    """
    results = random_background_replace_cv2_batch([frame])
    return results[0] if results else frame


def test_random_background_replace(video_path, output_path=None, max_frames=None, batch_size=16):
    """
    Test the random_background_replace_cv2 function by processing a video
    and saving the result with replaced backgrounds.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str, optional): Path to save the output video. 
                                     If None, generates automatically.
        max_frames (int, optional): Maximum number of frames to process.
                                   If None, processes all frames.
        batch_size (int, optional): Number of frames to process in each batch.
                                   Default is 16.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Validate input video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Information:")
    print(f"  - Resolution: {width}x{height}")
    print(f"  - FPS: {fps}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Batch size: {batch_size}")
    
    # Generate output path if not provided
    if output_path is None:
        video_dir = os.path.dirname(video_path)
        video_name = os.path.basename(video_path)
        video_name_no_ext = os.path.splitext(video_name)[0]
        output_path = os.path.join(video_dir, f"{video_name_no_ext}_bg_replaced.mp4")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"  - Output path: {output_path}")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video file: {output_path}")
        cap.release()
        return False
    
    # Read all frames first (or up to max_frames)
    frames_to_process = max_frames if max_frames else total_frames
    print(f"\nReading frames (max: {frames_to_process})...")
    
    frames_rgb = []
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Stop if max_frames reached
            if max_frames and frame_count > max_frames:
                break
            
            # Convert BGR to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_rgb.append(frame_rgb)
            
            # Print progress
            if frame_count % 50 == 0:
                print(f"  Read: {frame_count}/{frames_to_process} frames", end='\r')
        
        cap.release()
        print(f"\n  Total frames read: {len(frames_rgb)}")
        
        # Process frames in batches
        print(f"\nProcessing frames with batch size {batch_size}...")
        processed_frames = []
        
        for i in range(0, len(frames_rgb), batch_size):
            batch = frames_rgb[i:i+batch_size]
            processed_batch = random_background_replace_cv2_batch(batch)
            processed_frames.extend(processed_batch)
            
            # Print progress
            progress = (len(processed_frames) / len(frames_rgb)) * 100
            print(f"  Progress: {len(processed_frames)}/{len(frames_rgb)} frames ({progress:.1f}%)", end='\r')
        
        print(f"\n\nWriting output video...")
        
        # Write processed frames to output video
        for idx, processed_frame in enumerate(processed_frames):
            # Convert RGB to BGR for OpenCV
            processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
            out.write(processed_frame_bgr)
            
            if (idx + 1) % 50 == 0:
                print(f"  Written: {idx+1}/{len(processed_frames)} frames", end='\r')
        
        print(f"\n\nSuccessfully processed {len(processed_frames)} frames")
        print(f"Output saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if cap.isOpened():
            cap.release()
        out.release()
        print("Resources released")


def main():
    """Main function with example usage"""
    # Example usage - modify these paths as needed
    
    # Option 1: Use command line arguments
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
        output_video = sys.argv[2] if len(sys.argv) > 2 else None
        max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else None
        batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    else:
        # Option 2: Hardcoded paths for testing
        input_video = "D:/6. Datasets/Fall-Dataset/Cut/C00_010_0004-C1.mp4"
        output_video = "../outputs/test_bg_replaced.mp4"
        max_frames = 100  # Process only first 100 frames for quick testing
        batch_size = 16  # Process 16 frames at a time
    
    print("="*60)
    print("Testing random_background_replace_cv2 function (Batch Mode)")
    print("="*60)
    print(f"Input video: {input_video}")
    print()
    
    # Run the test
    success = test_random_background_replace(
        video_path=input_video,
        output_path=output_video,
        max_frames=max_frames,
        batch_size=batch_size
    )
    
    if success:
        print("\n✓ Test completed successfully!")
    else:
        print("\n✗ Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

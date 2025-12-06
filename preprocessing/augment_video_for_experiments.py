import cv2
import numpy as np
import os

# -----------------------------
# Augmentation functions
# -----------------------------

def augment_low_illumination(frame, alpha=0.7, beta=-40):
    """
    Simulate low illumination by darkening the frame.
    alpha < 1.0  -> reduces contrast
    beta < 0     -> shifts brightness down
    """
    # frame: uint8, OpenCV BGR
    # new_frame = alpha * frame + beta
    dark = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    return dark


def augment_low_resolution(frame, downscale_factor=0.5):
    """
    Simulate low-resolution CCTV by downscaling and upscaling the frame.
    downscale_factor: between (0, 1), e.g., 0.5 => half resolution.
    """
    h, w = frame.shape[:2]
    new_w = int(w * downscale_factor)
    new_h = int(h * downscale_factor)

    # Downscale
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Upscale back to original size
    low_res = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return low_res


def augment_partial_occlusion(frame, occ_ratio=0.2, position="random"):
    """
    Simulate partial occlusion by overlaying a solid rectangle (e.g., black).
    occ_ratio: fraction of frame occupied by the occlusion (area-wise approx).
    position: 'top', 'bottom', 'left', 'right', or 'random'
    """
    h, w = frame.shape[:2]
    area = h * w
    occ_area = int(area * occ_ratio)

    # Choose occlusion width/height (simple heuristic: square-ish block)
    occ_w = int(np.sqrt(occ_area))
    occ_h = int(np.sqrt(occ_area))

    # Clamp to frame size
    occ_w = min(occ_w, w)
    occ_h = min(occ_h, h)

    if position == "top":
        x1, y1 = 0, 0
    elif position == "bottom":
        x1, y1 = 0, h - occ_h
    elif position == "left":
        x1, y1 = 0, 0
    elif position == "right":
        x1, y1 = w - occ_w, 0
    else:  # random
        x1 = np.random.randint(0, w - occ_w + 1)
        y1 = np.random.randint(0, h - occ_h + 1)

    x2 = x1 + occ_w
    y2 = y1 + occ_h

    occluded = frame.copy()
    # Draw a dark rectangle (you can adjust color if needed)
    cv2.rectangle(occluded, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    return occluded


# -----------------------------
# Main processing function
# -----------------------------

def process_video(input_path,
                output_dir,
                low_illumination=True,
                low_resolution=True,
                partial_occlusion=True,
                show_progress=False,
            ):
    """
    Read a video and create augmented versions:
    - low illumination
    - low resolution
    - partial occlusion
    All saved as separate video files in output_dir.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Define VideoWriters for each augmentation
    writers = {}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or "XVID" depending on your setup

    if low_illumination:
        out_path_dark = os.path.join(output_dir, f"{base_name}_low_illumination.mp4")
        writers["dark"] = cv2.VideoWriter(out_path_dark, fourcc, fps, (width, height))

    if low_resolution:
        out_path_lowres = os.path.join(output_dir, f"{base_name}_low_resolution.mp4")
        writers["lowres"] = cv2.VideoWriter(out_path_lowres, fourcc, fps, (width, height))

    if partial_occlusion:
        out_path_occ = os.path.join(output_dir, f"{base_name}_partial_occlusion.mp4")
        writers["occ"] = cv2.VideoWriter(out_path_occ, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1

        # 1) Low illumination
        if "dark" in writers:
            dark = augment_low_illumination(frame, alpha=0.7, beta=-40)
            writers["dark"].write(dark)

        # 2) Low resolution
        if "lowres" in writers:
            lowres = augment_low_resolution(frame, downscale_factor=0.5)
            writers["lowres"].write(lowres)

        # 3) Partial occlusion
        if "occ" in writers:
            occ = augment_partial_occlusion(frame, occ_ratio=0.2, position="random")
            writers["occ"].write(occ)

        if show_progress and frame_count > 0 and current_frame % 50 == 0:
            print(f"Processed {current_frame}/{frame_count} frames")

    cap.release()
    for w in writers.values():
        w.release()

    print("Augmented videos saved in:", output_dir)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    base_path = "D:/6. Datasets/SPHAR-Dataset/unseen_abnormal_videos"
    output_folder = "D:/6. Datasets/SPHAR-Dataset/augmented_videos_experiments" 

    for video_file in os.listdir(base_path):
        if "augmented" in video_file:
            continue
        input_video = os.path.join(base_path, video_file)

        # randomly process each video with one augmentation
        augmentation_choices = ["low_illumination", "low_resolution", "partial_occlusion"]
        chosen_augmentation = np.random.choice(augmentation_choices)

        low_illumination = chosen_augmentation == "low_illumination"
        low_resolution = chosen_augmentation == "low_resolution"
        partial_occlusion = chosen_augmentation == "partial_occlusion"

        process_video(
            input_path=input_video,
            output_dir=output_folder,
            low_illumination=low_illumination,
            low_resolution=low_resolution,
            partial_occlusion=partial_occlusion,
            show_progress=True,
        )

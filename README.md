# Abnormal Human Action Detection using Deep Learning

A comprehensive deep learning system for real-time detection and classification of abnormal human actions in surveillance videos using transformer-based architectures and vision encoders.

## Project Overview

This project implements an automated abnormal action detection system designed for surveillance and security applications. The system processes video streams to identify abnormal human behaviors such as falling, hitting, kicking, stealing, vandalizing, and other suspicious activities, distinguishing them from normal activities like walking, sitting, and running.

## Key Features

- **Real-time Video Analysis**: Processes video streams frame-by-frame with sliding window approach for continuous monitoring
- **Binary Classification**: Classifies actions as either "normal" or "abnormal" with confidence scores
- **Multiple Vision Encoders**: Supports both CLIP (ViT-B/32) and ViT (Vision Transformer) for feature extraction
- **Transformer-Based Models**: Utilizes state-of-the-art transformer architecture with positional encoding for temporal sequence modeling
- **Flexible Configuration**: Adjustable window sizes, lag sampling rates, and abnormal detection thresholds
- **Video Augmentation**: Includes preprocessing tools for data augmentation to improve model robustness

## Technical Architecture

### 1. **Feature Extraction**
- **CLIP Encoder**: Uses OpenAI's CLIP ViT-B/32 model to extract 512-dimensional features from video frames
- **ViT Encoder**: Alternative Vision Transformer (vit_base_patch16_224) extracting 768-dimensional features
- Frame-level feature extraction with batch processing for GPU efficiency

### 2. **Temporal Modeling**
- **Transformer Model**: Custom transformer encoder with:
  - Learnable positional embeddings for sequence ordering
  - Multi-head attention mechanism (configurable heads)
  - GELU activation functions
  - Temporal average pooling for sequence aggregation
  - Differential features (frame-to-frame changes) concatenated with static features for enhanced temporal awareness

### 3. **Detection Pipeline**
- **Sliding Window Approach**: Maintains a window of recent frames (default: 120 frames)
- **Lag Sampling**: Samples frames within the window at configurable intervals (default: 5-6 stride)
- **Confidence Thresholding**: Aggregates predictions over time with configurable abnormal threshold (default: 5%)
- **Majority Voting**: Determines final classification based on accumulated predictions

## Dataset

The project uses the **SPHAR (Surveillance Perspective Human Action Recognition)** dataset containing 14 action classes:

**Normal Actions:**
- Sitting, Walking, Running, Neutral, Luggage

**Abnormal Actions:**
- Hitting, Kicking, Falling, Vandalizing, Panicking, Stealing, Murdering, Car Crash, Igniting

For more details, see the [dataset README](datasets/README.md).

## Project Structure

### Core Components

- **`utils/abnormal_action_detector.py`**: Main detection engine with real-time and batch processing capabilities
- **`notebooks/transformer_model.py`**: Custom transformer architecture implementation
- **`run_detection.py`**: Script for real-time video detection with visualization
- **`run_experiments.py`**: Batch evaluation script for model testing on unseen videos

### Preprocessing Tools (`preprocessing/`)

- **`extract_video_features_clip.py`**: Extracts CLIP features from videos
- **`extract_video_features_vit.py`**: Extracts ViT features from videos
- **`augment_video.py`**: Video augmentation for data enhancement
- **`select_video_subset.py`**: Dataset partitioning for train/val/test splits
- **`trim_video.py`**: Video trimming utilities
- **`extract_frames.py`**: Frame extraction from videos
- **`query_unseen_video.py`**: Tools for processing new/unseen videos

### Notebooks (`notebooks/`)

- **`transformer_model_training.ipynb`**: Training pipeline with early stopping and hyperparameter tuning
- **`lstm_model_training.ipynb`**: Alternative LSTM-based approach
- **`CLIP_image_encoder_inference.ipynb`**: CLIP feature extraction experiments
- **`data_preprocessing.ipynb`**: Data preparation and exploration
- **`llava_next_video_model_7b_inference.ipynb`**: Large language-vision model experiments

### Models

- **`transformer_model_v1.pt`**, **`v2.pt`**, **`v3.pt`**: Trained transformer models with different configurations
- Training histories and performance metrics stored in `transformer_train_histories.txt`

## Quick Start

### Prerequisites

```bash
pip install torch torchvision opencv-python clip timm pandas numpy matplotlib scikit-learn pillow
```

### Running Real-Time Detection

```python
python run_detection.py
```

Modify the `video_source` variable in the script to point to your video file.

### Running Batch Experiments

```python
python run_experiments.py
```

This will evaluate the model on unseen videos and generate results in the `outputs/` directory.

### Using the Detector in Your Code

```python
from utils.abnormal_action_detector import AbnormalActionDetector

# Initialize detector
detector = AbnormalActionDetector(
    model_path="models/transformer_model_v3.pt",
    window_size=120,
    lag_sampling=6,
    abnormal_threshold=0.05,
    image_encoder_type="clip"
)

# Detect abnormal actions
detector.detect_abnormal_action("path/to/video.mp4", is_save_result=False)

# Get classification results
class_idx, action_class, confidence = detector.get_abnormal_action_detection_results("path/to/video.mp4")
```

## Key Capabilities

### Real-Time Detection
- Processes video streams with visual feedback showing predictions and confidence scores
- Displays FPS performance metrics
- Color-coded alerts (red for abnormal, green for normal)
- Optional recording of annotated video output

### Batch Evaluation
- Processes multiple videos from datasets
- Generates classification reports with accuracy metrics
- Exports results to CSV for analysis
- Supports evaluation on unseen/test videos

### Configurable Parameters
- **Window size**: Number of frames to analyze (default: 120)
- **Lag sampling**: Frame sampling stride within window (5-6)
- **Abnormal threshold**: Percentage threshold for abnormal classification (5%)
- **Image encoder selection**: CLIP or ViT

## Performance Optimization

- **GPU Acceleration**: Leverages CUDA for faster inference
- **Batch Processing**: Processes frames in batches to optimize memory usage
- **Feature Caching**: Maintains sliding window of features in memory
- **Efficient Sampling**: Lag sampling reduces computational overhead while preserving temporal information

## Use Cases

- **Surveillance Systems**: Automated monitoring of public spaces, parking lots, and buildings
- **Security Applications**: Real-time detection of suspicious activities
- **Safety Monitoring**: Detecting falls and accidents in healthcare or industrial settings
- **Behavioral Analysis**: Understanding human action patterns in various contexts

## Technologies Used

- **Deep Learning**: PyTorch, Transformer architectures
- **Computer Vision**: OpenCV, CLIP, Vision Transformers (timm)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, scikit-learn metrics
- **Model Training**: Early stopping, dropout regularization, BCEWithLogitsLoss

## Model Architecture Details

The transformer model combines:
- **Input Embedding**: Projects concatenated static and differential features to model dimension
- **Positional Encoding**: Learnable embeddings to capture temporal ordering
- **Transformer Encoder**: Multiple layers of multi-head self-attention
- **Temporal Pooling**: Aggregates sequence information via mean pooling
- **Classification Head**: Linear layer producing binary logits

## Future Enhancements

- Multi-class abnormal action classification
- Integration with live camera feeds
- Mobile deployment optimization
- Ensemble model approaches
- Action localization in videos

## License

This project uses the SPHAR dataset. For dataset usage and licensing, refer to the [official SPHAR repository](https://github.com/AlexanderMelde/SPHAR-Dataset).

## Acknowledgments

- SPHAR Dataset creators for providing the surveillance action recognition dataset
- OpenAI for the CLIP model
- PyTorch and timm communities for excellent deep learning tools

import os
import pandas as pd
import cv2
import torch
from torchvision import transforms
from PIL import Image

def list_files_by_folder(folder_path, is_nested=False):
    files_data = []
    if os.path.exists(folder_path):
        for label in os.listdir(folder_path):  # First-level subfolders (e.g., "greeting")
            label_folder = os.path.join(folder_path, label)
            if os.path.isdir(label_folder):
                if is_nested:
                    # For nested structure (e.g., frames_sentence)
                    for subfolder in os.listdir(label_folder):  # Second-level subfolders
                        subfolder_path = os.path.join(label_folder, subfolder)
                        if os.path.isdir(subfolder_path):
                            for file in os.listdir(subfolder_path):
                                files_data.append({
                                    "file_path": os.path.join(subfolder_path, file),
                                    "label": label
                                })
                else:
                    # For non-nested structure
                    for file in os.listdir(label_folder):
                        files_data.append({
                            "file_path": os.path.join(label_folder, file),
                            "label": label
                        })
    else:
        print(f"Folder does not exist: {folder_path}")
    return files_data

def preprocess_image(file_path, image_size=(224, 224)):
    """
    Preprocess an image for PyTorch: resize, normalize, and convert to a tensor.
    """
    image = Image.open(file_path).convert("RGB")  # Open image and convert to RGB
    transform = transforms.Compose([
        transforms.Resize(image_size),  # Resize to desired dimensions
        transforms.ToTensor(),         # Convert to PyTorch tensor
        transforms.Normalize(          # Normalize with ImageNet mean/std values
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image)

def preprocess_videos(file_path, image_size=(224, 224), frame_count=16):
    """
    Preprocess a video: extract frames and apply transformations.
    """
    video_frames = []
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = torch.linspace(0, total_frames - 1, frame_count).long()

    for idx in range(total_frames):
        ret, frame = cap.read()
        if ret and idx in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame = Image.fromarray(frame)  # Convert to PIL Image
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            video_frames.append(transform(frame))

    cap.release()
    return torch.stack(video_frames)  # Combine all frames into a tensor

# Define paths
frames_sentence_path = "D:\\ISL_CSLRT_Corpus\\Frames_Sentence_Level"
frames_word_path = "D:\\ISL_CSLRT_Corpus\\Frames_Word_Level"
videos_sentence_path = "D:\\ISL_CSLRT_Corpus\\Videos_Sentence_Level"

# Collect data
sentence_frames = list_files_by_folder(frames_sentence_path, is_nested=True)
word_frames = list_files_by_folder(frames_word_path, is_nested=False)
sentence_videos = list_files_by_folder(videos_sentence_path, is_nested=False)

# Example usage: Preprocessing a single image and video
sample_image = sentence_frames[0]['file_path']
preprocessed_image = preprocess_image(sample_image)
print(f"Preprocessed Image Tensor Shape: {preprocessed_image.shape}")

sample_video = sentence_videos[0]['file_path']
preprocessed_video = preprocess_videos(sample_video)
print(f"Preprocessed Video Tensor Shape: {preprocessed_video.shape}")

# Create a DataFrame for metadata
all_data = sentence_frames + word_frames + sentence_videos
df = pd.DataFrame(all_data)
df.to_csv("metadata_pytorch.csv", index=False)

print("Metadata saved as metadata_pytorch.csv!")


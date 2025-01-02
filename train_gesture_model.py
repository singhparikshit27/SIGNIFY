import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
from torchvision.io import read_video
from PIL import Image
import pandas as pd
from tqdm import tqdm


# Step 1: Data Loader
class GestureDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_file, transform=None, num_frames=16):
        self.data = pd.read_csv(metadata_file)
        self.transform = transform
        self.num_frames = num_frames  # Number of frames to extract from videos

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx]['file_path']
        label = self.data.iloc[idx]['label']

        # Convert label to integer
        label_map = {label: i for i, label in enumerate(self.data['label'].unique())}
        label = label_map[label]

        if file_path.endswith(('.jpg', '.jpeg', '.png')):
            # Process image
            image = Image.open(file_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # Add a dummy dimension to treat the image as a single-frame video
            return image.unsqueeze(0), torch.tensor(label)

        elif file_path.endswith(('.mp4', '.MP4')):
            # Process video
            video_frames = self.process_video(file_path)
            return video_frames, torch.tensor(label)

        raise ValueError("Unsupported file type for training.")

    def process_video(self, file_path):
        # Read video and ensure proper frame count
        video, _, _ = read_video(file_path, pts_unit="sec")
    
        # Ensure video is in the expected format: (num_frames, height, width, channels)
        if len(video.shape) != 4 or video.shape[-1] != 3:
         raise ValueError(f"Unexpected video shape: {video.shape}. Expected shape: (num_frames, height, width, 3)")

        # Select up to `self.num_frames` frames evenly spaced from the video
        num_total_frames = video.shape[0]
        indices = torch.linspace(0, num_total_frames - 1, self.num_frames).long()
        selected_frames = video[indices]  # Select specific frames

        # Convert to PIL Images and apply transformations
        processed_frames = [
        self.transform(transforms.ToPILImage()(frame.permute(2, 0, 1)))  # Permute to (C, H, W) for transforms
        for frame in selected_frames
        ]
        return torch.stack(processed_frames)  # Return processed frames as a tensor



# Custom collate function for dataloader
def pad_collate_fn(batch):
    videos, labels = zip(*batch)
    
    # Find the maximum number of frames in the batch
    max_frames = max(video.size(0) for video in videos)
    
    # Pad videos to have the same number of frames
    padded_videos = []
    for video in videos:
        pad_frames = max_frames - video.size(0)
        padding = torch.zeros((pad_frames, *video.shape[1:]))
        padded_video = torch.cat([video, padding], dim=0)
        padded_videos.append(padded_video)
    
    return torch.stack(padded_videos), torch.tensor(labels)


# Step 2: Define Model
def create_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Step 3: Training Configuration
def train():
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Dataset
    dataset = GestureDataset("updated_metadata.csv", transform=image_transform, num_frames=16)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=pad_collate_fn)

    # Define Model
    num_classes = len(dataset.data['label'].unique())
    model = create_model(num_classes).to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for data, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data, labels = data.to(device), labels.to(device)

            # Flatten the video frames into the batch dimension
            batch_size, num_frames, channels, height, width = data.shape
            data = data.view(-1, channels, height, width)  # Combine frames into batch dimension
            labels = labels.repeat_interleave(num_frames)  # Repeat labels for frames to match data

            outputs = model(data)  # Model output with the new flattened batch size
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")
        scheduler.step()

    # Save Model
    torch.save(model.state_dict(), "gesture_model.pth")
    print("Model training complete and saved!")


# Run Training
if __name__ == "__main__":
    train()

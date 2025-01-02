from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

class GestureDataset(Dataset):
    def __init__(self, metadata_file, transform=None):
        self.data = pd.read_csv(metadata_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx]['file_path']
        label = self.data.iloc[idx]['label']
        
        if file_path.endswith('.jpg'):  # Handle image files
            image = Image.open(file_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label

        # Extend to handle videos later
        # if file_path.endswith('.mp4'):
        #     video_tensor = preprocess_videos(file_path)  # Use your function
        #     return video_tensor, label

# Define transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = GestureDataset("metadata_pytorch.csv", transform=image_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Test the dataloader
for batch in dataloader:
    images, labels = batch
    print(f"Batch images shape: {images.shape}, Batch labels: {labels}")
    break

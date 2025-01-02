import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.utils.data import Dataset
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the custom dataset class for your images
class GestureDataset(Dataset):
    def __init__(self, metadata_csv, transform=None):
        self.metadata = pd.read_csv(metadata_csv)
        self.transform = transform
        
        # Create a mapping from label names (strings) to integer labels
        self.label_map = {label: idx for idx, label in enumerate(self.metadata['label'].unique())}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.iloc[idx]['file_path']
        label = self.metadata.iloc[idx]['label']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Convert the string label to its corresponding integer value using the label_map
        label = self.label_map[label]
        
        # Ensure image and label are tensors
        image = torch.tensor(image) if not isinstance(image, torch.Tensor) else image
        label = torch.tensor(label) if not isinstance(label, torch.Tensor) else label
        
        return image, label


# Define the model
def create_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # Loading weights
    # Adjust the final layer to match the number of classes in your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Load the model
def load_model(model_path, num_classes):
    model = create_model(num_classes)
    # Load only the weights that match the model layers, ignoring the fc layer
    state_dict = torch.load(model_path, map_location=device)
    
    # Extract the weights for all layers except the fc layer
    state_dict = {k: v for k, v in state_dict.items() if k != 'fc.weight' and k != 'fc.bias'}
    
    # Load the weights
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)  # Move model to GPU or CPU
    return model

# Evaluate the model
def evaluate_model(model, dataloader, device):
    model.eval()
    true_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            data, labels = batch  # Unpack the batch
            
            # Check if data is a tuple of tensors, and convert if necessary
            if isinstance(data, tuple):
                data = data[0]  # Use only the first element of the tuple (image)
            
            # Ensure data and labels are tensors
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    
    return true_labels, predicted_labels

# Main script for loading and testing the model
def main():
    # Path to your metadata CSV
    metadata_path = 'test_metadata.csv'
    
    # Define any transformations you want to apply to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Initialize the dataset and dataloader
    test_dataset = GestureDataset(metadata_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load the model (adjust the num_classes based on your dataset)
    model_path = 'gesture_model.pth'
    num_classes = 26  # Adjust this to match your dataset
    model = load_model(model_path, num_classes)
    
    # Evaluate the model
    true_labels, predicted_labels = evaluate_model(model, test_dataloader, device)
    
    # Print or process the results
    print(f"True labels: {true_labels}")
    print(f"Predicted labels: {predicted_labels}")

if __name__ == "__main__":
    main()

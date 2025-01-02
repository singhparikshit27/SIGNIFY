import torch
import torch.nn as nn
import torchvision.models as models

# Define the ISL Model (Adjust based on your saved model architecture)
class ISLModel(nn.Module):
    def __init__(self, num_classes=215):  # Use 215 as in the saved model
        super(ISLModel, self).__init__()
        self.backbone = models.resnet18(pretrained=False)  # Assuming ResNet18
        self.backbone.fc = nn.Linear(512, num_classes)  # Match saved model

    def forward(self, x):
        return self.backbone(x)

# Function to load and adjust the model
def load_and_adjust_model(model_path, num_classes=3, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load the pre-trained model
    model = ISLModel(num_classes=215)  # Initialize with saved model's output size
    checkpoint = torch.load(model_path, map_location=device)

    # Load the state dict
    model.load_state_dict(checkpoint)
    print("Model loaded successfully!")

    # Adjust the final layer for the current task's number of classes
    model.backbone.fc = nn.Linear(512, num_classes)  # Adjust for 3 classes
    print(f"Adjusted model's output layer to {num_classes} classes.")

    # Move to device
    model = model.to(device)
    return model

# Main function to run the model
def main():
    # Define paths and settings
    model_path = "D:\ISL\gesture_model.pth"  # Replace with your model path
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load and adjust the model
    model = load_and_adjust_model(model_path, num_classes=3, device=device)

    # Simulate input (example)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Assuming input size is (3x224x224)

    # Perform a forward pass
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        output = model(dummy_input)
        print("Model output:", output)

if __name__ == "__main__":
    main()

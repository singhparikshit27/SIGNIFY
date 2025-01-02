import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bar

# Define your model (example)
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Example layers
        self.fc = nn.Linear(128, 3)  # Adjust for your input size and number of classes

    def forward(self, x):
        return self.fc(x)

# Define your dataset class (example)
class YourDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Ensure the data is of type float
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Labels are typically of dtype Long
        return data, label

# Evaluation function to get true and predicted labels

def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    predicted_labels = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        # Use tqdm to display the progress bar
        for data, labels in tqdm(dataloader, desc="Evaluating", ncols=100):
            data, labels = data.to(device), labels.to(device)  # Move to device (CPU/GPU)
            outputs = model(data)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the predicted labels
            true_labels.extend(labels.cpu().numpy())  # Move labels to CPU and convert to numpy
            predicted_labels.extend(predicted.cpu().numpy())  # Move predictions to CPU and convert to numpy

    return true_labels, predicted_labels


# Function to evaluate model performance with confusion matrix
def evaluate_model_performance(true_labels, predicted_labels):
    # Get unique classes
    classes = sorted(list(set(true_labels)))

    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate Accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    # Print Classification Report
    print("Classification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=[str(c) for c in classes]))

# Example of how to run the code
def main():
    # Set the device (use 'cuda' for GPU, 'cpu' for CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load your model
    model = YourModel().to(device)

    # Example data (replace with your actual data)
    data = [[1]*128] * 100  # Example data
    labels = [0] * 30 + [1] * 35 + [2] * 35  # Example labels (3 classes)

    # Create dataset and dataloader
    dataset = YourDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Get true and predicted labels
    true_labels, predicted_labels = evaluate_model(model, dataloader, device)

    # Evaluate the model performance (confusion matrix, accuracy, and classification report)
    evaluate_model_performance(true_labels, predicted_labels)

if __name__ == '__main__':
    main()

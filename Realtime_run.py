import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import pyttsx3
import mediapipe as mp
from PIL import Image

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model
def create_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Load the model with state_dict
def load_model(model_path, num_classes):
    model = create_model(num_classes)
    
    # Load the pretrained ResNet weights, ignoring the 'fc' layer
    state_dict = torch.load(model_path, map_location=device)
    
    # Remove the 'fc' weights from the checkpoint since it's incompatible
    state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
    
    # Load the remaining weights into the model
    model.load_state_dict(state_dict, strict=False)  # 'strict=False' allows the 'fc' layer mismatch
    
    # Reinitialize the fully connected layer to match the new number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    model.to(device)  # Move model to the appropriate device (GPU or CPU)
    model.eval()  # Set the model to evaluation mode
    return model

# Load the labels from a file (labels.txt or labels.json)
def load_labels(label_file):
    with open(label_file, 'r') as file:
        labels = file.read().splitlines()
    return labels

# Real-time gesture detection
def run_inference(model, labels):
    cap = cv2.VideoCapture(0)  # Start webcam capture
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Reduce detection and tracking confidence for speed
    mp_drawing = mp.solutions.drawing_utils
    engine = pyttsx3.init()  # Initialize text-to-speech engine
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing to a smaller size for faster inference
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to a smaller size for faster processing
        frame_resized = cv2.resize(frame, (640, 480))  # Smaller resolution for faster processing
        
        # Convert the resized frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Perform hand tracking
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Convert the landmarks to an image tensor (skipping drawing landmarks for speed)
                hand_image = frame_resized  # Using the resized frame
                hand_image_pil = Image.fromarray(cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB))

                # Apply transformations to the image
                hand_image_tensor = transform(hand_image_pil).unsqueeze(0).to(device)

                # Get predictions from the model
                with torch.no_grad():
                    outputs = model(hand_image_tensor)
                    _, predicted = torch.max(outputs, 1)
                
                predicted_class = predicted.item()

                # Get the predicted label from the loaded labels
                predicted_label = labels[predicted_class]
                print(f"Predicted Gesture: {predicted_label}")

                # Speak out the prediction
                engine.say(predicted_label)
                engine.runAndWait()

        # Display the resized frame (without landmarks for now to reduce lag)
        cv2.imshow("Hand Gesture Recognition", frame_resized)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to load the model and run inference
def main():
    model_path = 'gesture_model.pth'  # Adjust to your model path
    label_file = 'labels.txt'  # File containing class labels (one per line)
    
    # Load the labels
    labels = load_labels(label_file)
    num_classes = len(labels)  # Number of classes is the length of the labels file
    model = load_model(model_path, num_classes)

    # Start the real-time inference
    run_inference(model, labels)

if __name__ == "__main__":
    main()

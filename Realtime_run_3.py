import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pyttsx3
import mediapipe as mp
import threading
from queue import Queue

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create a lightweight model
def create_model(num_classes):
    model = models.mobilenet_v2(weights='DEFAULT')
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

# Load the model with state_dict
def load_model(model_path, num_classes):
    model = create_model(num_classes)
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

# Load the labels from a file
def load_labels(label_file):
    with open(label_file, 'r') as file:
        labels = file.read().splitlines()
    return labels

# Real-time inference thread
def inference_thread(frame_queue, model, labels):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    engine = pyttsx3.init()

    while True:
        if not frame_queue.empty():
            hand_image = frame_queue.get()
            hand_image_pil = Image.fromarray(cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB))
            hand_image_tensor = transform(hand_image_pil).unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                outputs = model(hand_image_tensor)
                _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
            predicted_label = labels[predicted_class]
            print(f"Predicted Gesture: {predicted_label}")

            # Speak out the prediction
            engine.say(predicted_label)
            engine.runAndWait()

# Real-time video capture and hand detection
def run_camera(model, labels):
    cap = cv2.VideoCapture(0)

    # Set decent resolution and frame rate
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, 
                           max_num_hands=1, 
                           min_detection_confidence=0.5, 
                           min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    frame_queue = Queue(maxsize=1)  # Queue to pass frames to the inference thread

    # Start inference thread
    threading.Thread(target=inference_thread, args=(frame_queue, model, labels), daemon=True).start()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Crop region of interest (ROI) around the hand
                height, width, _ = frame.shape
                x_min = int(min(landmark.x for landmark in landmarks.landmark) * width)
                y_min = int(min(landmark.y for landmark in landmarks.landmark) * height)
                x_max = int(max(landmark.x for landmark in landmarks.landmark) * width)
                y_max = int(max(landmark.y for landmark in landmarks.landmark) * height)

                roi = frame[max(0, y_min - 20):min(height, y_max + 20),
                            max(0, x_min - 20):min(width, x_max + 20)]

                # Pass the cropped ROI to the inference queue
                if not frame_queue.full():
                    frame_queue.put(roi)

        cv2.imshow("Hand Gesture Recognition", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    model_path = 'gesture_model.pth'
    label_file = 'labels.txt'
    labels = load_labels(label_file)
    num_classes = len(labels)
    model = load_model(model_path, num_classes)

    run_camera(model, labels)

if __name__ == "__main__":
    main()

import cv2
import speech_recognition as sr
import pyttsx3
import time
import os

# Function to capture audio and convert it to text
def capture_audio_and_convert_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for your speech...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio)
        print(f"Recognized Text: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results from Google Speech Recognition.")
        return None

# Map Text to Sign Language Gesture Images or Videos
def text_to_sign_language(text):
    gesture_mapping = {
        'hello': 'hello_gesture.jpg',  # Image for gesture 'hello'
        'thank you': 'thank_you_gesture.jpg',  # Image for gesture 'thank you'
        # Add more words and corresponding gesture images or videos
    }
    
    # Return the gesture image corresponding to the recognized text
    return gesture_mapping.get(text.lower(), None)

# Function to display sign language gestures
def show_sign_language_gesture(gesture_image_path):
    gesture = cv2.imread(gesture_image_path)
    if gesture is not None:
        cv2.imshow("Sign Language Gesture", gesture)
        cv2.waitKey(2000)  # Display the gesture for 2 seconds
    else:
        print("Gesture not found!")

# Main function to run the speech-to-sign language conversion
def main():
    while True:
        # Capture and convert speech to text
        text = capture_audio_and_convert_to_text()

        if text:
            # Map the text to the corresponding sign language gesture
            gesture_image = text_to_sign_language(text)

            if gesture_image:
                # Show the gesture corresponding to the recognized text
                show_sign_language_gesture(gesture_image)
            else:
                print("No gesture found for the recognized text.")

        # Break the loop after a certain period of time or user input
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import os
import pandas as pd

def list_files_by_folder(folder_path, is_nested=False):
    files_data = []
    if os.path.exists(folder_path):
        print(f"Checking if folder exists: {folder_path} - {os.path.exists(folder_path)}")  # Debugging line
        print(f"Scanning folder: {folder_path}")  # Debugging line
        for label in os.listdir(folder_path):  # First-level subfolders (e.g., "greeting")
            label_folder = os.path.join(folder_path, label)
            if os.path.isdir(label_folder):
                if is_nested:
                    # For nested structure (e.g., frames_sentence)
                    for subfolder in os.listdir(label_folder):  # Second-level subfolders (e.g., "sub1", "sub2")
                        subfolder_path = os.path.join(label_folder, subfolder)
                        if os.path.isdir(subfolder_path):
                            print(f"Scanning subfolder: {subfolder_path}")  # Debugging line
                            for file in os.listdir(subfolder_path):  # Images/videos
                                print(f"Found file: {file}")  # Debugging line
                                files_data.append({
                                    "file_path": os.path.join(subfolder_path, file),  # Full path to the file
                                    "label": label  # Top-level folder as the label (e.g., "greeting")
                                })
                else:
                    # For non-nested structure (e.g., frames_word, videos_sentence)
                    for file in os.listdir(label_folder):  # Images/videos
                        print(f"Found file: {file}")  # Debugging line
                        files_data.append({
                            "file_path": os.path.join(label_folder, file),  # Full path to the file
                            "label": label  # Top-level folder as the label (e.g., "hello")
                        })
    else:
        print(f"Folder does not exist: {folder_path}")  # Debugging line
    return files_data

# Define the folder paths (replace these with your actual paths)
frames_sentence_path = r"D:\ISL_CSLRT_Corpus\Frames_Sentence_Level"
frames_word_path = r"D:\ISL_CSLRT_Corpus\Frames_Word_Level"
videos_sentence_path = r"D:\ISL_CSLRT_Corpus\Videos_Sentence_Level"

# Collect data from each folder
sentence_frames = list_files_by_folder(frames_sentence_path, is_nested=True)  # Nested folder structure for sentence
word_frames = list_files_by_folder(frames_word_path, is_nested=False)  # Non-nested structure for word
sentence_videos = list_files_by_folder(videos_sentence_path, is_nested=False)  # Non-nested structure for videos

# Combine all data
all_data = sentence_frames + word_frames + sentence_videos

# Create a Pandas DataFrame and save as CSV
df = pd.DataFrame(all_data)
df.to_csv("metadata1.csv", index=False)

# Print summary of collected files
print(f"Total files found: {len(all_data)}")
print("Dataset organized and metadata saved as metadata.csv!")

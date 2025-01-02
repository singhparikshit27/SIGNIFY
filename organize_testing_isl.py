import os
import pandas as pd

def generate_metadata(dataset_path, output_csv):
    # List to store file paths and labels
    metadata = []

    # Traverse the dataset directory
    for class_label in os.listdir(dataset_path):
        class_folder = os.path.join(dataset_path, class_label)
        
        # Skip if not a directory
        if not os.path.isdir(class_folder):
            continue
        
        for filename in os.listdir(class_folder):
            file_path = os.path.join(class_folder, filename)
            
            # Only include image files
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                metadata.append({'file_path': file_path, 'label': class_label})
    
    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_csv, index=False)
    print(f"Metadata saved to {output_csv}")

# Example usage
generate_metadata(r"D:\ISL_Dataset", r"D:\ISL_Dataset\test_metadata.csv")


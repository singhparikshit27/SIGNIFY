import pandas as pd

# Step 1: Read the CSV file
metadata_df = pd.read_csv('metadata_pytorch.csv')

# Step 2: Filter out rows where 'file_path' ends with '.db'
filtered_df = metadata_df[~metadata_df['file_path'].str.endswith('.db')]

# Step 3: Save the updated DataFrame back to a CSV file
filtered_df.to_csv('updated_metadata.csv', index=False)

print("Removed all .db file entries and saved to 'updated_metadata.csv'.")

import pandas as pd

# Load the Excel file
df = pd.read_excel(r"C:\Users\DELL\Desktop\metadata.xlsx")

# Assume the column with data is named 'Data'
# Remove text before the comma (including the comma)
df['Cleaned_Data'] = df['Data'].apply(lambda x: x.split(",", 1)[1] if "," in x else x)

# Save the cleaned data to a .txt file
df['Cleaned_Data'].to_csv("labels.txt", index=False, header=False)
import pandas as pd
import json
import math

# Read the CSV file
file_path = 'datasets/dataset-q1.csv'
df = pd.read_csv(file_path, header=None, names=["current", "latest", "root", "label"])

# Function to preprocess text if needed
def preprocess_text(current, latest, root, label):
    if current is None or (isinstance(current, float) and math.isnan(current)):
        current = ""
    if latest is None or (isinstance(latest, float) and math.isnan(latest)):
        latest = ""
    if root is None or (isinstance(root, float) and math.isnan(root)):
        root = ""
    prompt = f"""Current User Text: {current}\n Latest User Text: {latest}\n Root User Text: {root}\n Does the current user's post / latest user's post / root user's post express an attitude towards the economic situation, government management, and/or personal life? Directly reply with Yes or No. Answer: {label}"""
    return prompt

# Convert the dataframe into the required JSON format
data = []
for index, row in df.iterrows():
    entry = {
        "text": preprocess_text(row["current"], row["latest"], row["root"], row["label"]),
        "label": row["label"]
    }
    data.append(entry)

# Save the processed data into a JSON file
output_file = 'datasets/test-dataset-q2.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Data successfully saved to {output_file}")
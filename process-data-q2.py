import pandas as pd
import json
import math

# Read the CSV file
file_path = 'datasets/50k.csv'
# df = pd.read_csv(file_path, header=None, names=["current", "label"])
df = pd.read_csv(file_path, header=None, names=["weibo_id", "current", "label"])

# Function to preprocess text if needed
def preprocess_text(current, label):
    if current is None or (isinstance(current, float) and math.isnan(current)):
        current = ""
    # prompt = f"""Current User Text: {current}\n. Does the current user's post express an attitude towards the economic situation, government management, or personal life? Directly reply with Yes or No.""" # Answer: {label}
    # prompt = f"{current}"
    prompt = (
        f"Classify the following text into 'Yes' or 'No'. Answer 'Yes' if the text expresses an attitude towards the Chinese economic situation, "
        f"government management, or personal life. Answer 'No' otherwise. Text: \"{current}\". Provide the response in the format: Answer: Yes or Answer: No. Do not write anything else."
    )
    return prompt

# Convert the dataframe into the required JSON format
data = []
for index, row in df.iterrows():
    entry = {
        "text": preprocess_text(row["current"], row["label"]),
        "label": "Yes" if row["label"] == 1.0 else "No", 
        "weibo_id": str(row["weibo_id"])
    }
    data.append(entry)

# Save the processed data into a JSON file
output_file = 'datasets/test-50k.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Data successfully saved to {output_file}")
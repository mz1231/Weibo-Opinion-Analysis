from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import Accelerator
from datasets import load_dataset
import torch
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Initialize the accelerator
accelerator = Accelerator()

# Load the pre-trained Qwen-2 model and tokenizer with GPU support
model_name = "Qwen/Qwen2-7B-Instruct"  # Replace with the actual model name if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, local_files_only=True)

# Move model to the appropriate device(s)
model = accelerator.prepare(model)

# Load Dataset
print("Loading dataset")
test_data = load_dataset('json', data_files='datasets/test-50k.json', split='train[50%:]')
print("Loaded successfully")

# Initialize the text generation pipeline with GPU support
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# Function to evaluate a piece of text and return "Yes" or "No"
def evaluate_text(text):
    # Define the refined prompt
    prompt = (
        f"Classify the following text into 'Yes' or 'No'. Answer 'Yes' if the text expresses an attitude towards the Chinese economic situation, "
        f"government management, or personal life. Answer 'No' otherwise. Text: \"{text}\". Provide the response in the format: Answer: Yes or Answer: No. Do not write anything else."
    )
    
    # Generate the response with reduced max_new_tokens for faster processing
    result = generator(prompt, max_new_tokens=10, num_return_sequences=1, truncation=True)
    generated_text = result[0]['generated_text'].strip().lower()
    answer_part = generated_text.split("answer:")[-1].strip()
    # Extract and format the answer
    if 'yes' in answer_part:
        return "Yes"
    elif 'no' in answer_part:
        return "No"
    else:
        return "Unknown"

# Lists to store predictions and true labels
predictions = []
true_labels = []

# Lists for output files:
false_positives = []
false_negatives = []
true_positives = []
true_negatives = []
false_predictions = []
correct_predictions = []
yes_guessed = []
no_guessed = []

for i in range(len(test_data)):
    text = test_data[i]['text']
    label = test_data[i]['label']
    weibo_id = test_data[i]['weibo_id']
    answer = evaluate_text(text)
    
    # Store predictions and true labels
    predictions.append(answer)
    true_labels.append(label)
    if answer != label:
        false_predictions.append(weibo_id)
        if answer == 'Yes':
            false_positives.append(weibo_id)
            yes_guessed.append(weibo_id)
        elif answer == 'No':
            false_negatives.append(weibo_id)
            no_guessed.append(weibo_id)
    else:
        correct_predictions.append(weibo_id)
        if answer == 'Yes':
            true_positives.append(weibo_id)
            yes_guessed.append(weibo_id)
        elif answer == 'No':
            true_negatives.append(weibo_id)
            no_guessed.append(weibo_id)

# Save lists to text files
with open('qwen2-eval2/false_predictions.txt', 'w') as f:
    for item in false_predictions:
        f.write(f"{item}\n")

with open('qwen2-eval2/correct_predictions.txt', 'w') as f:
    for item in correct_predictions:
        f.write(f"{item}\n")

with open('qwen2-eval2/false_positives.txt', 'w') as f:
    for item in false_positives:
        f.write(f"{item}\n")

with open('qwen2-eval2/false_negatives.txt', 'w') as f:
    for item in false_negatives:
        f.write(f"{item}\n")

with open('qwen2-eval2/true_negatives.txt', 'w') as f:
    for item in true_negatives:
        f.write(f"{item}\n")

with open('qwen2-eval2/true_positives.txt', 'w') as f:
    for item in true_positives:
        f.write(f"{item}\n")

with open('qwen2-eval2/yes_guessed.txt', 'w') as f:
    for item in yes_guessed:
        f.write(f"{item}\n")

with open('qwen2-eval2/no_guessed.txt', 'w') as f:
    for item in no_guessed:
        f.write(f"{item}\n")


# Calculate confusion matrix and classification report
cm = confusion_matrix(true_labels, predictions, labels=["Yes", "No"])
report = classification_report(true_labels, predictions, labels=["Yes", "No"])

# Print confusion matrix
print("Confusion Matrix:")
print(cm)

# Print classification report in the desired format
print("\nTest Classification Report:")
print(report)

print("\nWrong predictions:")
print(false_predictions)

print("\nFalse Positives:")
print(false_positives)

print("\nFalse Negatives:")
print(false_negatives)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# Initialize the accelerator
accelerator = Accelerator()

print("Attempting to load tokenizer and model")
saved_model = "models/llama3-q2-trial2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(saved_model, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(saved_model, torch_dtype=torch.bfloat16, local_files_only=True)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

print("Loaded successfully")
# Move model to the appropriate device(s)
model, tokenizer = accelerator.prepare(model, tokenizer)

print("Loading dataset")
test_data = load_dataset('json', data_files='datasets/test-dataset-q2.json', split='train[90%:]')
print("Loaded successfully")

# Tokenize the test dataset
def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding="max_length", max_length=512)

test_data = test_data.map(tokenize_function, batched=True)
test_data = test_data.rename_column("label", "labels")

test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Prepare dataloader
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1)
test_dataloader = accelerator.prepare(test_dataloader)

# Evaluation
model.eval()
predictions, true_labels = [], []

for batch in test_dataloader:
    with torch.no_grad():
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    logits = outputs.logits
    predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    true_labels.extend(batch['labels'].cpu().numpy())

# Calculate precision, recall, accuracy, and F1 score
precision = precision_score(true_labels, predictions, average='binary')
recall = recall_score(true_labels, predictions, average='binary')
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='binary')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")

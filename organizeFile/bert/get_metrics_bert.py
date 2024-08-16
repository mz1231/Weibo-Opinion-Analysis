import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    logging
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import numpy as np
import torch.nn.functional as F
from dataset_utils import load_and_split_dataset, load_and_upsample_dataset

# Set environment variable to disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure the Hugging Face cache directory is set
os.environ['TRANSFORMERS_CACHE'] = '/scratch/network/mz1231/.cache/huggingface'
os.environ['HF_HOME'] = '/scratch/network/mz1231/.cache/huggingface'

# Model from Hugging Face hub
base_model = "models/bert-chinese-finetuned"

# Load tokenizer and model from the local cache
tokenizer = BertTokenizer.from_pretrained(base_model, local_files_only=True)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load model
model = BertForSequenceClassification.from_pretrained(base_model, num_labels=2)

# Set pad_token_id for the model
model.config.pad_token_id = tokenizer.pad_token_id

print("Loaded model and tokenizer")

dataset, test_df, train_df = load_and_split_dataset('datasets/sequence-dataset-q2.json')
print(f"Train dataset length: {len(dataset['train'])}")
print(f"Val dataset length: {len(dataset['val'])}")
print(f"Test dataset length: {len(dataset['test'])}")

# Manually adjusted class weights
class_weights = torch.tensor([0.1, 0.9], dtype=torch.float32)
print(f"Class weights: {class_weights}")

# Tokenize datasets
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

# Apply the tokenization function to the datasets
tokenized_data = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_data.set_format("torch")

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)
    return {
        'balanced_accuracy': balanced_accuracy_score(labels, predictions),
        'accuracy': accuracy_score(labels, predictions)
    }

# Generate predictions and evaluate
def generate_predictions(model, test_set):
    sentences = test_set.text.tolist()
    batch_size = 32
    all_outputs = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            all_outputs.append(outputs.logits)
        
    final_outputs = torch.cat(all_outputs, dim=0)
    test_set['predictions'] = final_outputs.argmax(axis=1).cpu().numpy()

generate_predictions(model, train_df)

def get_metrics_result(test_set):
    y_test = test_set.label
    y_pred = test_set.predictions
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

get_metrics_result(train_df)

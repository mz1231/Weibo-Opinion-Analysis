import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import transformers
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
    Trainer
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, classification_report
import torch.nn.functional as F
import numpy as np

# Set environment variable to disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the accelerator
accelerator = Accelerator()

# Ensure the Hugging Face cache directory is set
os.environ['TRANSFORMERS_CACHE'] = '/scratch/network/mz1231/.cache/huggingface'
os.environ['HF_HOME'] = '/scratch/network/mz1231/.cache/huggingface'

# Model from Hugging Face hub
base_model = "meta-llama/Meta-Llama-3-8B"

# Fine-tuned model
new_model = "models/llama3-q2-trial3"

# 4-bit quantization configuration
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load tokenizer and model from the local cache
tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    num_labels=2  # Set the number of labels for binary classification
)

# Set pad_token_id for the model
model.config.pad_token_id = tokenizer.pad_token_id

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loaded model and tokenizer")

# Load dataset from JSON file
dataset = load_dataset('json', data_files='datasets/sequence-dataset-q2.json')
df = pd.DataFrame(dataset['train'])

# Splitting the dataframe into separate dataframes based on the labels
label_1_df = df[df['label'] == 0]
label_2_df = df[df['label'] == 1]

len_label_1 = len(label_1_df)  # Length of label 0 dataframe
len_label_2 = len(label_2_df)  # Length of label 1 dataframe

# Shuffle each label dataframe
label_1_df = label_1_df.sample(frac=1).reset_index(drop=True)
label_2_df = label_2_df.sample(frac=1).reset_index(drop=True)

# 80% train, 10% val, 10% test
train_split_ratio = 0.8
val_split_ratio = 0.1
test_split_ratio = 0.1

# For label 0
label_1_train_size = int(len_label_1 * train_split_ratio)
label_1_val_size = int(len_label_1 * val_split_ratio)
label_1_test_size = len_label_1 - label_1_train_size - label_1_val_size

# For label 1
label_2_train_size = int(len_label_2 * train_split_ratio)
label_2_val_size = int(len_label_2 * val_split_ratio)
label_2_test_size = len_label_2 - label_2_train_size - label_2_val_size

# Split dataset
label_1_train = label_1_df.iloc[:label_1_train_size]
label_1_val = label_1_df.iloc[label_1_train_size:label_1_train_size + label_1_val_size]
label_1_test = label_1_df.iloc[label_1_train_size + label_1_val_size:]

label_2_train = label_2_df.iloc[:label_2_train_size]
label_2_val = label_2_df.iloc[label_2_train_size:label_2_train_size + label_2_val_size]
label_2_test = label_2_df.iloc[label_2_train_size + label_2_val_size:]

# Concatenating the splits back together
train_df = pd.concat([label_1_train, label_2_train]).sample(frac=1).reset_index(drop=True)
val_df = pd.concat([label_1_val, label_2_val]).sample(frac=1).reset_index(drop=True)
test_df = pd.concat([label_1_test, label_2_test]).sample(frac=1).reset_index(drop=True)

# Converting pandas DataFrames into Hugging Face Dataset objects
dataset_train = Dataset.from_pandas(train_df)
dataset_val = Dataset.from_pandas(val_df)
dataset_test = Dataset.from_pandas(test_df)

print(f"Train dataset length: {len(dataset_train)}")
print(f"Val dataset length: {len(dataset_val)}")
print(f"Test dataset length: {len(dataset_test)}")

# Combine them into a single DatasetDict
dataset = DatasetDict({
    'train': dataset_train,
    'val': dataset_val,
    'test': dataset_test
})

# class_weights = (1/train_df.label.value_counts(normalize=True).sort_index()).tolist()
# class_weights = torch.tensor(class_weights)
# class_weights = class_weights/class_weights.sum()

# Manually adjusted class weights
class_weights = torch.tensor([0.1, 0.9], dtype=torch.float32)
print(f"Class weights: {class_weights}")

print(class_weights)
# Tokenize datasets
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

# Apply the tokenization function to the datasets
tokenized_data = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_data.set_format("torch")

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# PEFT parameters
peft_params = LoraConfig(
    lora_alpha=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="SEQ_CLS",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_params)

print("Loaded PEFT params")

# Training parameters
training_params = TrainingArguments(
    output_dir="models/checkpoints-q2-trial3",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    learning_rate=1e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    hub_token=os.getenv("HF_TOKEN")
)

print("Loaded training params")

# Custom trainer with class weights
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.get('logits')
        if self.class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        else:
            loss = F.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(evaluations):
    predictions, labels = evaluations
    predictions = np.argmax(predictions, axis=1)
    return {
        'balanced_accuracy': balanced_accuracy_score(labels, predictions),
        'accuracy': accuracy_score(labels, predictions)
    }

# Initialize the trainer
trainer = CustomTrainer(
    model=model,
    args=training_params,
    train_dataset=tokenized_data['train'],
    eval_dataset=tokenized_data['val'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    class_weights=class_weights.tolist(),
    compute_metrics=compute_metrics
)

print("Set the trainer")

# Prepare the trainer with the accelerator
trainer = accelerator.prepare(trainer)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

# Generate predictions and evaluate
def generate_predictions(model, df_test):
    sentences = df_test.text.tolist()
    batch_size = 32
    all_outputs = []

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i + batch_size]
        inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            all_outputs.append(outputs['logits'])
        
    final_outputs = torch.cat(all_outputs, dim=0)
    df_test['predictions'] = final_outputs.argmax(axis=1).cpu().numpy()

generate_predictions(model, test_df)

def get_metrics_result(test_df):
    y_test = test_df.label
    y_pred = test_df.predictions
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

get_metrics_result(test_df)

import os
import torch
from datasets import load_dataset, Dataset
import json
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator
from tensorboard import notebook
from dataset_utils import upsample_dataset

# Initialize the accelerator
accelerator = Accelerator()

# Ensure the Hugging Face cache directory is set
os.environ['TRANSFORMERS_CACHE'] = '/scratch/network/mz1231/.cache/huggingface'
os.environ['HF_HOME'] = '/scratch/network/mz1231/.cache/huggingface'

# Model from Hugging Face hub
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"

# Fine-tuned model
new_model = "models/llama3-q1-upsampling3"

# 4-bit quantization configuration
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load tokenizer and model from the local cache
tokenizer = transformers.AutoTokenizer.from_pretrained(
    base_model, 
    local_files_only=True)
model = transformers.AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map= "auto",      #{"": 0},
    torch_dtype=torch.bfloat16, 
    local_files_only=True)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loaded model and tokenizer")

# Load dataset from JSON file
train_dataset = load_dataset('json', data_files='datasets/dataset-q1.json', split='train[:80%]')
val_dataset = load_dataset('json', data_files='datasets/dataset-q1.json', split='train[80%:90%]')
print(f"First entry of train dataset: {train_dataset[0]}")
print(f"Size of train data: {len(train_dataset)}")
print(f"First entry of validation dataset: {val_dataset[0]}")
print(f"Size of validation data: {len(val_dataset)}")
print("Loaded dataset")

# Upsample the training dataset using the function from dataset_utils.py
upsampled_dataset = upsample_dataset(train_dataset)
print(f"Size of upsampled train data: {len(upsampled_dataset)}")

# PEFT parameters
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

print("loaded PEFT params")
# training parameters
training_params = TrainingArguments(
    output_dir="models/checkpoints-q1-upsampling3",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=500,
    eval_steps=500,  # Add this line to set evaluation interval
    evaluation_strategy="steps",  # enable periodic evaluation
    learning_rate=1e-3,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

print("loaded training params")
trainer = SFTTrainer(
    model=model,
    train_dataset=upsampled_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_params,
    dataset_text_field="text",
    max_seq_length=1024,  # Directly passing the max_seq_length here
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)
print("Set the trainer")

# Prepare the trainer with the accelerator
trainer = accelerator.prepare(trainer)

trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)
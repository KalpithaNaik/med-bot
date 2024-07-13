import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
torch.backends.mps.enabled = False
torch.backends.cudnn.enabled = False

import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)

# Define paths
model_name = "gpt2"  # Base model
train_file = "/Users/kalpithanaik/Desktop/medbot/att3.py/training_data/train.txt"
output_dir = "/Users/kalpithanaik/Desktop/medbot/att3.py/fine-tuned-model"

# Set device
device = torch.device("cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
def load_dataset_from_file(file_path):
    return load_dataset('text', data_files={'train': file_path})

train_dataset = load_dataset_from_file(train_file)['train']

# Tokenize dataset
def tokenize_function(examples):
    result = tokenizer(examples['text'], return_special_tokens_mask=True, truncation=True, padding='max_length', max_length=128)
    return {k: torch.tensor(v).to(device) for k, v in result.items()}

tokenized_datasets = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    save_steps=10_000,
    save_total_limit=2,
    use_cpu=True  # Force CPU usage
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

print(f"Number of samples in train_dataset: {len(train_dataset)}")
print(f"Sample from dataset: {train_dataset[0]}")

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
import fitz  # PyMuPDF
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import os
import torch

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set device to CPU
device = torch.device("cpu")

# Define functions for PDF processing and text cleaning
def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file.
    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        str: Extracted text content from the PDF.
    """
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except FileNotFoundError:
        print(f"File not found: {pdf_path}")
        return ""

def clean_text(text):
    """Preprocesses text by removing irrelevant content and normalizing format.
    Args:
        text (str): Text to be cleaned.
    Returns:
        str: Cleaned text.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Define paths to your PDFs (replace with actual paths)
pdf_paths = [
    "/Users/kalpithanaik/Desktop/medbot/att3.py/data/Oxford_American_Handbook_of_Cardiology_Oxford_American_Handbooks.pdf",
    "/Users/kalpithanaik/Desktop/medbot/att3.py/data/Oxford_Handbook_of_Emergency_Medicine_Jonathan_P_Wyatt,_Robin_N.pdf",
    "/Users/kalpithanaik/Desktop/medbot/att3.py/data/Oxford_Handbook_of_Respiratory_Medicine_3rd_Ed_–_Oxford_University.pdf"
]

# Preprocess text from each PDF
preprocessed_text_files = []
for pdf_path in pdf_paths:
    print(f"Processing: {pdf_path}")
    extracted_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(extracted_text)
    with open(f"preprocessed_pdf{pdf_paths.index(pdf_path)+1}.txt", "w") as file:
        file.write(cleaned_text)
    preprocessed_text_files.append(f"preprocessed_pdf{pdf_paths.index(pdf_path)+1}.txt")

# Combine preprocessed text into a single file
combined_text = ""
for filename in preprocessed_text_files:
    with open(filename, "r") as file:
        combined_text += file.read()

# Tokenize the combined text
model_name = "gpt2"  # Choose a suitable language model (e.g., gpt2, distilgpt2)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the pad token to be the eos token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

tokenized_text = tokenizer(combined_text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

# Move model and inputs to CPU
model.to(device)
tokenized_text = {k: v.to(device) for k, v in tokenized_text.items()}

# Create a dataset from the tokenized text
dataset = Dataset.from_dict({
    "input_ids": tokenized_text["input_ids"].tolist(),
    "attention_mask": tokenized_text["attention_mask"].tolist()
})

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust training epochs as needed
    per_device_train_batch_size=4,  # Adjust batch size based on your CPU memory
    save_steps=10_000,
    save_total_limit=2,
    no_cuda=True  # Disable CUDA
)

# Create Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the model
trainer.train()
print("Training complete! Your language model is saved in the 'results' directory.")

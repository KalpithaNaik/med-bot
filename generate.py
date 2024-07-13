from transformers import pipeline

# Define paths
model_name = "/Users/kalpithanaik/Desktop/medbot/att3.py/fine-tuned-model"

# Load the fine-tuned model for generation
generator = pipeline("text-generation", model=model_name, tokenizer=model_name)

# Generate text based on a prompt
generated_text = generator("Hello, how are you?", max_length=100, num_return_sequences=1)

print(generated_text)

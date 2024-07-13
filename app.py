from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model_name = "gpt2"  # Path to your fine-tuned model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.json.get('text')
    if input_text is None:
        return jsonify({'error': 'No input text provided'}), 400
    
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_length=100,  # Increase max_length for longer responses
        num_return_sequences=1,
        num_beams=5,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2  # Adjust repetition_penalty
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(port=5001)  # Use a different port, e.g., 5001

from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from googletrans import Translator
from gtts import gTTS
import pandas as pd
import torch
import os

app = Flask(__name__)

# Load the model and tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load your dataset for symptom checking
csv_path = "/Users/kalpithanaik/Desktop/medbot/att3.py/training_data/train.csv"
df = pd.read_csv(csv_path)

# Initialize translator
translator = Translator()

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

# Symptom checker function
def check_symptoms(symptoms):
    inputs = tokenizer(symptoms, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

    predicted_condition = df.loc[predicted_label, 'condition']
    
    return predicted_condition

# Route for handling symptom checker requests
@app.route('/symptom-checker', methods=['POST'])
def symptom_checker():
    try:
        user_input = request.json.get('symptoms')
        predicted_condition = check_symptoms(user_input)
        return jsonify({'condition': predicted_condition})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for handling medication reminders (placeholder)
@app.route('/medication-reminder', methods=['POST'])
def medication_reminder():
    try:
        data = request.json
        medication = data.get('medication')
        time = data.get('time')
        return jsonify({'message': f'Reminder set for {medication} at {time}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for translating text
@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.json
        text = data.get('text')
        target_language = data.get('target_language', 'en')
        
        translation = translator.translate(text, dest=target_language)
        
        return jsonify({'translation': translation.text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route for text-to-speech conversion
@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        text = data.get('text')
        language_code = data.get('language_code', 'en')
        
        tts = gTTS(text=text, lang=language_code)
        output_file = "output.mp3"
        tts.save(output_file)

        return jsonify({'message': f'Audio content written to file "{output_file}"'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Emergency contacts (dummy data)
emergency_contacts = {
    'police': '911',
    'fire': '911',
    'ambulance': '911',
    'doctor': 'Your doctor\'s contact information'
}

# Route for fetching emergency contacts
@app.route('/emergency-contacts', methods=['GET'])
def get_emergency_contacts():
    return jsonify(emergency_contacts)

# Interactive health assessment (dummy questions)
health_questions = {
    'fever': {
        'question': 'Do you have a fever?',
        'options': ['Yes', 'No']
    },
    'headache': {
        'question': 'Do you have a headache?',
        'options': ['Yes', 'No']
    }
    # Add more questions as needed
}

# Route for interactive health assessment
@app.route('/health-assessment', methods=['POST'])
def health_assessment():
    try:
        data = request.json
        responses = data.get('responses')

        health_status = {}

        for question_id, answer in responses.items():
            health_status[question_id] = {
                'question': health_questions[question_id]['question'],
                'answer': answer
            }

        # Implement logic to assess health based on responses

        return jsonify({'health_status': health_status})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001)
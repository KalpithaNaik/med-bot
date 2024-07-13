import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from googletrans import Translator
from gtts import gTTS
import pandas as pd
import torch
import os
import base64

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    return tokenizer, model

tokenizer, model = load_model()

# Load your dataset for symptom checking
@st.cache_data
def load_dataset():
    csv_path = "/Users/kalpithanaik/Desktop/medbot/att3.py/training_data/train.csv"
    return pd.read_csv(csv_path)

df = load_dataset()

# Initialize translator
translator = Translator()

# Symptom checker function
def check_symptoms(symptoms):
    inputs = tokenizer(symptoms, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
    predicted_condition = df.loc[predicted_label, 'condition']
    return predicted_condition

# Emergency contacts (dummy data)
emergency_contacts = {
    'police': '911',
    'fire': '911',
    'ambulance': '911',
    'doctor': 'Your doctor\'s contact information'
}

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

# Streamlit app
def main():
    st.title("Medical Chatbot")

    # Sidebar for feature selection
    feature = st.sidebar.selectbox(
        "Choose a feature",
        ["Symptom Checker", "Medication Reminder", "Translator", "Text-to-Speech", "Emergency Contacts", "Health Assessment"]
    )

    if feature == "Symptom Checker":
        st.header("Symptom Checker")
        symptoms = st.text_input("Describe your symptoms:")
        if st.button("Check Symptoms"):
            if symptoms:
                condition = check_symptoms(symptoms)
                st.write(f"Based on your symptoms, you may have: {condition}")
            else:
                st.warning("Please enter your symptoms.")

    elif feature == "Medication Reminder":
        st.header("Medication Reminder")
        medication = st.text_input("Enter medication name:")
        time = st.time_input("Set reminder time:")
        if st.button("Set Reminder"):
            st.success(f"Reminder set for {medication} at {time}")

    elif feature == "Translator":
        st.header("Translator")
        text = st.text_area("Enter text to translate:")
        target_language = st.selectbox("Select target language:", ["en", "es", "fr", "de", "it"])
        if st.button("Translate"):
            if text:
                translation = translator.translate(text, dest=target_language)
                st.write("Translation:", translation.text)
            else:
                st.warning("Please enter text to translate.")

    elif feature == "Text-to-Speech":
        st.header("Text-to-Speech")
        text = st.text_area("Enter text for speech synthesis:")
        language_code = st.selectbox("Select language:", ["en", "es", "fr", "de", "it"])
        if st.button("Generate Speech"):
            if text:
                tts = gTTS(text=text, lang=language_code)
                tts.save("output.mp3")
                audio_file = open("output.mp3", "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                os.remove("output.mp3")
            else:
                st.warning("Please enter text for speech synthesis.")

    elif feature == "Emergency Contacts":
        st.header("Emergency Contacts")
        for service, number in emergency_contacts.items():
            st.write(f"{service.capitalize()}: {number}")

    elif feature == "Health Assessment":
        st.header("Health Assessment")
        responses = {}
        for question_id, question_data in health_questions.items():
            response = st.radio(question_data['question'], question_data['options'])
            responses[question_id] = response
        if st.button("Submit Assessment"):
            st.write("Your responses:")
            for question_id, answer in responses.items():
                st.write(f"{health_questions[question_id]['question']} {answer}")
            st.write("Please consult a healthcare professional for a proper diagnosis.")

if __name__ == "__main__":
    main()
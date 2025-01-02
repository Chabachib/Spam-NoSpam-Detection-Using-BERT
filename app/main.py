from classifier import *
from utils import *
import streamlit as st

st.title("Multilingual Text Spam Classification")

# language selection
st.sidebar.header("Settings")
src_lang = st.sidebar.selectbox("Select source language", ["Arabic", "French", "English", "Spanish"])

st.subheader("Input Text")
user_text = st.text_area("Enter the text to classify", height=200)

if st.button("Classify"):
    if not user_text.strip():
        st.warning("Please enter text for classification.")
    else:
        st.text("Loading models...")

        # Load translation models
        translation_models = {
            "Arabic": load_translation_model("ar", "en"),
            "French": load_translation_model("fr", "en"),
            "Spanish": load_translation_model("es", "en"),
        }

        # Load the BERT classifier
        bert_model_path = f"../models/bert_classifier.pth"  # Update this path
        bert_model = load_custom_bert_classifier(bert_model_path)
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Handle English directly, otherwise translate
        if src_lang == "English":
            st.subheader("Input Text (No Translation Required)")
            st.text(user_text)
            text_to_classify = user_text
        else:
            # Translate the input text
            translator_tokenizer, translator_model = translation_models[src_lang]
            translated_text = translate_text(user_text, translator_tokenizer, translator_model)
            st.subheader("Translated Text")
            st.text(translated_text)
            text_to_classify = translated_text

        # Classify the translated text
        prediction = predict_spam(translated_text, bert_model, bert_tokenizer)
        st.subheader("Prediction")
        st.success(prediction)

import streamlit as st
import torch
from app.classifier import *
from transformers import BertModel, BertTokenizer, MarianMTModel, MarianTokenizer

# Load translation models
@st.cache_resource
def load_translation_model(src_lang, tgt_lang):
    model_name = rf"models/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Load the BERT classifier
@st.cache_resource
def load_custom_bert_classifier(
    model_path, bert_model_name="bert-base-uncased", num_classes=2
):
    model = BERTClassifier(bert_model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model


def translate_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text
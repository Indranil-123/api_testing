import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
from langdetect import detect
from googletrans import Translator
from textblob import TextBlob
from typing import Tuple
from src.chat_module.serializer import TextInput, IntentResponse

# ✅ Load dataset and model globally
with open('src/chat_module/dataset/intent_training_dataset.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

model = BertForSequenceClassification.from_pretrained('codegof/PAVA-M1')
tokenizer = BertTokenizer.from_pretrained('codegof/PAVA-M1')

with open('src/chat_module/encoder/label_encoder.json', 'r') as f:
    labels = json.load(f)

model.eval()

# ✅ Synchronous language transform
def language_transform(sentence: str) -> str:
    translator = Translator()
    translated = translator.translate(sentence, dest='en')
    return translated.text

# ✅ Synchronous spelling correction
def correct_spelling(sentence: str) -> str:
    blob = TextBlob(sentence)
    corrected_sentence = blob.correct()
    return str(corrected_sentence)

# ✅ Synchronous transformation function
def transformation(sentence: str) -> Tuple[str, str]:
    code = detect(sentence)

    lang_map = {
        'hi': 'Hindi',
        'bn': 'Bengali',
        'ta': 'Tamil',
        'te': 'Telugu',
        'en': 'English'
    }

    predicted_language = lang_map.get(code, 'Other')

    if predicted_language == 'English':
        corrected = correct_spelling(sentence)
        return corrected, predicted_language
    else:
        transformed = language_transform(sentence)
        return transformed, predicted_language

# ✅ Main intent prediction logic
def predict_intent(input_text: TextInput) -> IntentResponse:
    sentence = input_text.text

    transformed_text, detected_language = transformation(sentence)

    # Tokenize and predict
    inputs = tokenizer(transformed_text, return_tensors='pt', truncation=True, padding='max_length', max_length=32)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)

    # Safe label retrieval
    try:
        intent = labels[predicted_class.item()]
    except (IndexError, KeyError):
        intent = "dont_know"

    response_row = df[df['intent'] == intent]

    if confidence.item() < 0.6 or response_row.empty:
        return IntentResponse(
            intent="dont_know",
            confidence=confidence.item(),
            response="I'm sorry, I don't understand your query clearly.",
            detected_language=detected_language
        )
    else:
        response = response_row['response'].iloc[0]
        return IntentResponse(
            intent=intent,
            confidence=confidence.item(),
            response=response,
            detected_language=detected_language
        )

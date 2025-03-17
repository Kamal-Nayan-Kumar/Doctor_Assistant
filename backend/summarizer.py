import requests

HUGGINGFACE_API_TOKEN = "your api key"
MODEL_NAME = "facebook/bart-large-cnn"

def summarize_text(text):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payload = {"inputs": text}
    response = requests.post(f"https://api-inference.huggingface.co/models/{MODEL_NAME}", headers=headers, json=payload)
    return response.json()[0]['summary_text']

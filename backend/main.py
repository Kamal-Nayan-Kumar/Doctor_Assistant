from flask import Flask, request, jsonify
from flask_cors import CORS
from whisper_transcribe import transcribe_audio
from chatbot import chatbot_response
from summarizer import summarize_text
from ocr_extraction import extract_text

app = Flask(__name__)
CORS(app)

# Route for speech-to-text (Whisper)
@app.route('/transcribe', methods=['POST'])
def transcribe():
    audio_file = request.files['file']
    text = transcribe_audio(audio_file)
    return jsonify({"transcription": text})

# Route for chatbot (Hugging Face)
@app.route('/chatbot', methods=['POST'])
def chat():
    user_message = request.json.get("message")
    response = chatbot_response(user_message)
    return jsonify({"reply": response})

# Route for document summarization (Hugging Face)
@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.json.get("text")
    summary = summarize_text(text)
    return jsonify({"summary": summary})

# Route for OCR (Tesseract.js)
@app.route('/extract-text', methods=['POST'])
def ocr():
    image_file = request.files['image']
    extracted_text = extract_text(image_file)
    return jsonify({"text": extracted_text})

if __name__ == '__main__':
    app.run(debug=True)

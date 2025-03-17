from huggingface_hub import InferenceClient
import os
from tenacity import retry, stop_after_attempt, wait_exponential

# Use the provided Hugging Face API key (Remove if using .env)
HUGGINGFACE_API_TOKEN = "your api key"  # Replace with your key
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

# Initialize Hugging Face Client
client = InferenceClient(model=MODEL_NAME, token=HUGGINGFACE_API_TOKEN)

def format_mistral_prompt(user_message: str) -> str:
    """Formats prompt for Mistral instruct model"""
    return f"<s>[INST] {user_message} [/INST]"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_with_retry(prompt: str) -> str:
    """Retry mechanism for API calls with better error handling"""
    try:
        response = client.text_generation(
            prompt=prompt,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            do_sample=True,
        )

        # ✅ Debugging: Print raw response
        print("Raw Hugging Face Response:", response)

        # ✅ Check if response is valid before extracting text
        if not response or not isinstance(response, str):
            return "Error: Invalid response from Hugging Face API"

        # ✅ Ensure safe text extraction
        cleaned_response = response.split("[/INST]")[-1].strip() if "[/INST]" in response else response.strip()
        
        return cleaned_response

    except Exception as e:
        return f"Error: {str(e)}"

def chatbot_response(user_message: str) -> str:
    """Processes user message and returns chatbot response"""
    try:
        formatted_prompt = format_mistral_prompt(user_message)
        return generate_with_retry(formatted_prompt)
    except Exception as e:
        return f"Error: {str(e)}"

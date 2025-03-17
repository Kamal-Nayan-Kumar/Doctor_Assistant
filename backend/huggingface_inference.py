import requests
import os

# Set your Hugging Face API token
api_token = "your api key"

# Set the model name
model_name = "meta-llama/Llama-2-7b"

# Function to make API call
def query_model(prompt):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)
    print("Status Code:", response.status_code)
    print("Raw Response:", response.text)
    return response.json()


# Get user input
user_prompt = input("Enter your prompt: ")

# Make the API call
result = query_model(user_prompt)

# Print the result
print(result)

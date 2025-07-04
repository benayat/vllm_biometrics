import time

from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam
import base64
from io import BytesIO
from PIL import Image

# Initialize OpenAI client for local server
client = OpenAI(
    base_url="http://10.100.102.24:8000/v1",
    api_key="none"
)

# Define model name
model_name = "gemma-3-27b-it"

# Load and prepare image
image1_path = "data/Avner1.jpg"  # Replace with your image path
image2_path = "data/Inon2.jpg"

def encode_image_to_base64(image_path)-> str:
    image = Image.open(image_path, mode='r')
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

image1_base64 = encode_image_to_base64(image1_path)
image2_base64 = encode_image_to_base64(image2_path)
user_message: ChatCompletionUserMessageParam = {
    "role": "user",
    "content": [
        ChatCompletionContentPartTextParam(
            type="text",
            text="Are these two people likely to be related by family? Analyze the faces and explain your reasoning."
        ),
        ChatCompletionContentPartImageParam(
            type="image_url",
            image_url={
                "url": f"data:image/png;base64,{image1_base64}"
            }
        ),
        ChatCompletionContentPartImageParam(
            type="image_url",
            image_url={
                "url": f"data:image/png;base64,{image2_base64}"
            }
        )
    ]
}

start_time = time.time()
response = client.chat.completions.create(
    model=model_name,
    messages=[user_message],
    temperature=0.2,
    max_tokens=512
)
elapsed = time.time() - start_time

print("RESPONSE:", response.choices[0].message.content)
print(f"Request took {elapsed:.2f} seconds.")

if hasattr(response, "usage"):
    usage = response.usage
    print(f"Prompt tokens: {usage.prompt_tokens}, Completion tokens: {usage.completion_tokens}, Total tokens: {usage.total_tokens}")
else:
    print("Token usage information not available in response.")
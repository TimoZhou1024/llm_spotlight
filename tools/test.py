import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["INCEPTION_API_KEY"],
    base_url="https://api.inceptionlabs.ai/v1"
)

response = client.chat.completions.create(
    model="mercury-2",
    messages=[{"role": "user", "content": "What is a diffusion model?"}],
    max_tokens=1000
)
print(response.choices[0].message.content)
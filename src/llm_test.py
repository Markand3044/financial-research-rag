import os 

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

question = "what is the purpose of an annual report ?"

response = client.chat.completions.create(
    model = "qwen/qwen3.6-27b",
    messages=[
        {
            "role":"system",
            "content":"you are a helpful financial research assistance."
        },
        {
            "role":"user",
            "content":question
        }
    ],
    reasoning_format="hidden"
)

print("\n --- LLM ANSWER --- \n")
print(response.choices[0].message.content)
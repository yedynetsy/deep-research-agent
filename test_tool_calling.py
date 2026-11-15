
import os

from dotenv import load_dotenv
from groq import Groq

from agent.tools import TOOL_SCHEMAS


load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an AI research assistant. "
                "Use the available tools to research topics. "
                "Do not invent sources or URLs."
            )
        },
        {
            "role": "user",
            "content": (
                "Find recent research about reducing "
                "hallucinations in RAG systems."
            )
        }
    ],
    tools=TOOL_SCHEMAS,
    tool_choice="required",
    temperature=0.2,
)

message = response.choices[0].message

print("Content:", message.content)
print("Tool calls:")

for tool_call in message.tool_calls or []:
    print("Function:", tool_call.function.name)
    print("Arguments:", tool_call.function.arguments)

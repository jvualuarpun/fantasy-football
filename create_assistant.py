from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)

assistant = client.beta.assistants.create(
    name="Fantasy Football Assistant",
    instructions="You are a Discord bot assistant that remembers Discord messages and helps with fantasy football.",
    model="gpt-4o-mini"   # ✅ correct model name
)

print("Assistant ID:", assistant.id)
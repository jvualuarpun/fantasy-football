from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)

# Create a new Thread
thread = client.beta.threads.create()
print("Thread ID:", thread.id)
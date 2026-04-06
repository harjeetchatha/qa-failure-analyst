from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class FailurePayload(BaseModel):
    test_name: str
    error_message: str
    stack_trace: str

@app.post("/analyze")
async def analyze_failure(payload: FailurePayload):
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="You are a QA expert. Analyze Playwright test failures and provide a clear diagnosis and suggested fix.",
        messages=[
            {
                "role": "user",
                "content": f"""
                Test Name: {payload.test_name}
                Error: {payload.error_message}
                Stack Trace: {payload.stack_trace}
                
                Diagnose this failure and suggest a fix.
                """
            }
        ]
    )
    return {
        "test_name": payload.test_name,
        "diagnosis": message.content[0].text
    }

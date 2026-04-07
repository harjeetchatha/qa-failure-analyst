from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
from fastembed import TextEmbedding
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
engine = create_engine(os.getenv("DATABASE_URL"))
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

class FailurePayload(BaseModel):
    test_name: str
    error_message: str
    stack_trace: str

def retrieve_similar_patterns(query: str, top_k: int = 3):
    query_embedding = list(embedding_model.embed([query]))[0].tolist()
    embedding_str = str(query_embedding)
    
    with engine.connect() as conn:
        results = conn.execute(text("""
            SELECT pattern_name, error_type, description, solution,
                   1 - (embedding <=> CAST(:embedding AS vector)) as similarity
            FROM failure_patterns
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """), {
            "embedding": embedding_str,
            "top_k": top_k
        })
        return results.fetchall()

@app.post("/analyze")
async def analyze_failure(payload: FailurePayload):
    query = f"{payload.test_name} {payload.error_message}"
    similar_patterns = retrieve_similar_patterns(query)
    
    context = ""
    for pattern in similar_patterns:
        context += f"""
        Pattern: {pattern.pattern_name}
        Error Type: {pattern.error_type}
        Description: {pattern.description}
        Solution: {pattern.solution}
        Similarity: {round(pattern.similarity * 100)}%
        ---
        """
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="""You are a QA expert. Analyze Playwright test failures using the similar patterns provided.
        Base your diagnosis primarily on the retrieved patterns.
        If the patterns are relevant, cite them in your diagnosis.
        Always provide a specific fix.""",
        messages=[
            {
                "role": "user",
                "content": f"""
                Test Name: {payload.test_name}
                Error: {payload.error_message}
                Stack Trace: {payload.stack_trace}
                
                Similar patterns from knowledge base:
                {context}
                
                Diagnose this failure and suggest a fix based on the patterns above.
                """
            }
        ]
    )
    
    return {
        "test_name": payload.test_name,
        "diagnosis": message.content[0].text,
        "similar_patterns": [
            {
                "pattern": p.pattern_name,
                "similarity": f"{round(p.similarity * 100)}%"
            } for p in similar_patterns
        ]
    }
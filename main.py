from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
from fastembed import TextEmbedding
from sqlalchemy import create_engine, text
from github import Github
from dotenv import load_dotenv
import os
import json

load_dotenv()

app = FastAPI()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
engine = create_engine(os.getenv("DATABASE_URL"))
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")
gh = Github(os.getenv("GITHUB_TOKEN"))

class FailurePayload(BaseModel):
    test_name: str
    error_message: str
    stack_trace: str
    repo: str | None = None
    pr_number: int | None = None

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

def calculate_confidence(patterns):
    if not patterns:
        return "low", 0
    top_score = round(patterns[0].similarity * 100)
    if top_score >= 75:
        return "high", top_score
    elif top_score >= 50:
        return "medium", top_score
    else:
        return "low", top_score

def post_pr_comment(repo_name: str, pr_number: int, comment: str):
    try:
        repo = gh.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        pr.create_issue_comment(comment)
        return True
    except Exception as e:
        print(f"Could not post PR comment: {e}")
        return False

def format_pr_comment(diagnosis: dict) -> str:
    confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    emoji = confidence_emoji.get(diagnosis["confidence"], "⚪")

    patterns_text = ""
    for p in diagnosis["similar_patterns"]:
        patterns_text += f"\n  - {p['pattern']} ({p['similarity']} match)"

    return f"""## 🤖 QA Failure Analyst

**Test:** `{diagnosis["test_name"]}`
**Confidence:** {emoji} {diagnosis["confidence"].upper()} ({diagnosis["confidence_score"]}%)

### Root Cause
{diagnosis["root_cause"]}

### Suggested Fix
{diagnosis["fix"]}

### Matched Patterns
{patterns_text}

---
*Diagnosed automatically by QA Failure Analyst*"""

@app.post("/analyze")
async def analyze_failure(payload: FailurePayload):
    query = f"{payload.test_name} {payload.error_message}"
    similar_patterns = retrieve_similar_patterns(query)
    confidence_level, confidence_score = calculate_confidence(similar_patterns)

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
        system="""You are a QA expert. Analyze Playwright test failures.
        You must respond with ONLY a JSON object in this exact format:
        {
            "root_cause": "one sentence describing the root cause",
            "fix": "specific code or steps to fix the issue"
        }
        No extra text. No markdown. Just the JSON object.""",
        messages=[
            {
                "role": "user",
                "content": f"""
                Test Name: {payload.test_name}
                Error: {payload.error_message}
                Stack Trace: {payload.stack_trace}

                Similar patterns from knowledge base:
                {context}

                Return the JSON diagnosis.
                """
            }
        ]
    )

    raw = message.content[0].text.strip()
    claude_response = json.loads(raw)

    diagnosis = {
        "test_name": payload.test_name,
        "confidence": confidence_level,
        "confidence_score": confidence_score,
        "root_cause": claude_response["root_cause"],
        "fix": claude_response["fix"],
        "similar_patterns": [
            {
                "pattern": p.pattern_name,
                "similarity": f"{round(p.similarity * 100)}%"
            } for p in similar_patterns
        ],
        "auto_posted_to_pr": False
    }

    if payload.repo and payload.pr_number:
        comment = format_pr_comment(diagnosis)
        posted = post_pr_comment(payload.repo, payload.pr_number, comment)
        diagnosis["auto_posted_to_pr"] = posted

    return diagnosis
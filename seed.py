from fastembed import TextEmbedding
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))
embedding_model = TextEmbedding("BAAI/bge-small-en-v1.5")

patterns = [
    {
        "pattern_name": "Timeout on element click",
        "error_type": "TimeoutError",
        "description": "locator.click timeout exceeded waiting for selector",
        "solution": "Add waitForLoadState before click. Check if element is hidden behind overlay. Verify selector still exists in DOM."
    },
    {
        "pattern_name": "Element not found",
        "error_type": "LocatorError",
        "description": "strict mode violation, locator resolved to multiple elements or no elements found",
        "solution": "Use more specific selector. Add data-testid attributes. Check if element is inside iframe."
    },
    {
        "pattern_name": "Network request timeout",
        "error_type": "NetworkError",
        "description": "API call timeout or network request failed during test execution",
        "solution": "Increase timeout for slow API endpoints. Add retry logic. Mock the API call in tests."
    },
    {
        "pattern_name": "Authentication failure",
        "error_type": "AuthError",
        "description": "Login failed or session expired during test run",
        "solution": "Use storageState to reuse authenticated sessions. Check credentials in environment variables."
    },
    {
        "pattern_name": "Async operation not awaited",
        "error_type": "AsyncError",
        "description": "Test completed before async operation finished causing false positive or missed assertion",
        "solution": "Add await keyword before async calls. Use waitForResponse or waitForRequest to wait for network calls."
    },
    {
        "pattern_name": "Screenshot assertion failed",
        "error_type": "VisualError",
        "description": "Visual comparison failed because screenshot does not match baseline",
        "solution": "Update baseline screenshots with --update-snapshots flag. Check for dynamic content like dates or animations."
    },
    {
        "pattern_name": "Test data conflict",
        "error_type": "DataError",
        "description": "Test failed because data from previous test run was not cleaned up",
        "solution": "Add beforeEach cleanup. Use unique test data per run with timestamps. Implement proper teardown."
    },
    {
        "pattern_name": "CI environment difference",
        "error_type": "EnvironmentError",
        "description": "Test passes locally but fails in CI pipeline",
        "solution": "Check for hardcoded localhost URLs. Ensure environment variables are set in CI. Use headless mode."
    },
    {
        "pattern_name": "Race condition",
        "error_type": "RaceCondition",
        "description": "Flaky test that passes and fails intermittently due to timing issues",
        "solution": "Replace fixed timeouts with waitForSelector or waitForResponse. Add explicit waits for state changes."
    },
    {
        "pattern_name": "SOAP note save failure",
        "error_type": "FormError",
        "description": "Form submission failed silently or validation error not caught in provider portal",
        "solution": "Wait for success toast or confirmation element after submit. Check for validation error messages before asserting success."
    }
]

print("Loading embedding model...")

with engine.connect() as conn:
    conn.execute(text("DELETE FROM failure_patterns"))
    print("Seeding failure patterns...")
    
    for pattern in patterns:
        text_to_embed = f"{pattern['pattern_name']} {pattern['description']} {pattern['solution']}"
        embedding = list(embedding_model.embed([text_to_embed]))[0].tolist()
        
        conn.execute(text("""
            INSERT INTO failure_patterns (pattern_name, error_type, description, solution, embedding)
            VALUES (:name, :error_type, :description, :solution, :embedding)
        """), {
            "name": pattern["pattern_name"],
            "error_type": pattern["error_type"],
            "description": pattern["description"],
            "solution": pattern["solution"],
            "embedding": str(embedding)
        })
        print(f"✓ Seeded: {pattern['pattern_name']}")
    
    conn.commit()

print("Done. 10 patterns seeded.")
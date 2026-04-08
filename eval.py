from dotenv import load_dotenv
import anthropic
import os
import json

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

test_cases = [
    {
        "question": "Playwright test failing with TimeoutError waiting for submit button selector",
        "ground_truth": "The submit button is not visible or clickable. Add waitForLoadState and waitForSelector before clicking.",
        "contexts": [
            "Timeout on element click: locator.click timeout exceeded waiting for selector. Solution: Add waitForLoadState before click. Check if element is hidden behind overlay.",
            "Race condition: Flaky test that passes and fails intermittently. Solution: Replace fixed timeouts with waitForSelector or waitForResponse."
        ]
    },
    {
        "question": "Playwright test failing because element not found strict mode violation",
        "ground_truth": "The locator is matching multiple elements or no elements. Use a more specific selector with data-testid attributes.",
        "contexts": [
            "Element not found: strict mode violation, locator resolved to multiple elements. Solution: Use more specific selector. Add data-testid attributes.",
            "CI environment difference: Test passes locally but fails in CI. Solution: Check for hardcoded localhost URLs."
        ]
    },
    {
        "question": "Authentication failed during Playwright test run session expired",
        "ground_truth": "The session expired during the test. Use storageState to reuse authenticated sessions across tests.",
        "contexts": [
            "Authentication failure: Login failed or session expired during test run. Solution: Use storageState to reuse authenticated sessions.",
            "Test data conflict: Test failed because data from previous run was not cleaned up. Solution: Add beforeEach cleanup."
        ]
    },
    {
        "question": "Playwright test passes locally but fails in CI pipeline",
        "ground_truth": "Environment difference between local and CI. Check for hardcoded localhost URLs and ensure environment variables are set in CI.",
        "contexts": [
            "CI environment difference: Test passes locally but fails in CI pipeline. Solution: Check for hardcoded localhost URLs. Ensure environment variables are set in CI.",
            "Async operation not awaited: Test completed before async operation finished. Solution: Add await keyword before async calls."
        ]
    },
    {
        "question": "Playwright test flaky intermittently passing and failing race condition",
        "ground_truth": "Race condition due to timing issues. Replace fixed timeouts with waitForSelector or waitForResponse.",
        "contexts": [
            "Race condition: Flaky test that passes and fails intermittently due to timing issues. Solution: Replace fixed timeouts with waitForSelector or waitForResponse.",
            "Async operation not awaited: Test completed before async operation finished. Solution: Add await keyword before async calls."
        ]
    }
]

def get_answer(question: str, contexts: list) -> str:
    context_text = "\n".join(contexts)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="You are a QA expert. Answer based only on the provided context.",
        messages=[{
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context_text}\n\nDiagnose based only on the context above."
        }]
    )
    return message.content[0].text

def evaluate_answer(question: str, answer: str, contexts: str, ground_truth: str) -> dict:
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="""You are an evaluation expert. Score the answer on three metrics.
        Return ONLY a JSON object like this:
        {
            "faithfulness": 0.9,
            "relevancy": 0.8,
            "context_recall": 0.7,
            "reasoning": "one sentence explanation"
        }
        Scores are between 0 and 1. No extra text.""",
        messages=[{
            "role": "user",
            "content": f"""
            Question: {question}
            Context provided: {contexts}
            Answer given: {answer}
            Ground truth: {ground_truth}
            
            Score these three metrics:
            - faithfulness: did the answer come from the context or was it made up?
            - relevancy: did the answer actually address the question?
            - context_recall: does the context contain enough info to answer correctly?
            
            Return only the JSON object.
            """
        }]
    )
    return json.loads(message.content[0].text.strip())

print("Running QA Failure Analyst Evaluation")
print("="*50)

scores = {
    "faithfulness": [],
    "relevancy": [],
    "context_recall": []
}

for i, case in enumerate(test_cases):
    print(f"\nTest case {i+1}/{len(test_cases)}: {case['question'][:50]}...")
    
    answer = get_answer(case["question"], case["contexts"])
    context_text = "\n".join(case["contexts"])
    
    eval_result = evaluate_answer(
        case["question"],
        answer,
        context_text,
        case["ground_truth"]
    )
    
    scores["faithfulness"].append(eval_result["faithfulness"])
    scores["relevancy"].append(eval_result["relevancy"])
    scores["context_recall"].append(eval_result["context_recall"])
    
    print(f"  Faithfulness:    {eval_result['faithfulness']:.2f}")
    print(f"  Relevancy:       {eval_result['relevancy']:.2f}")
    print(f"  Context Recall:  {eval_result['context_recall']:.2f}")
    print(f"  Reasoning:       {eval_result['reasoning']}")

avg_faithfulness = sum(scores["faithfulness"]) / len(scores["faithfulness"])
avg_relevancy = sum(scores["relevancy"]) / len(scores["relevancy"])
avg_context_recall = sum(scores["context_recall"]) / len(scores["context_recall"])
overall = (avg_faithfulness + avg_relevancy + avg_context_recall) / 3

print("\n" + "="*50)
print("FINAL EVALUATION RESULTS")
print("="*50)
print(f"Faithfulness:     {avg_faithfulness:.2f}")
print(f"Relevancy:        {avg_relevancy:.2f}")
print(f"Context Recall:   {avg_context_recall:.2f}")
print(f"Overall Score:    {overall:.2f}")
print("="*50)

threshold = 0.8
print("\nQUALITY GATES")
print(f"Faithfulness:   {'PASSED' if avg_faithfulness >= threshold else 'FAILED'} (threshold: {threshold})")
print(f"Relevancy:      {'PASSED' if avg_relevancy >= threshold else 'FAILED'} (threshold: {threshold})")
print(f"Context Recall: {'PASSED' if avg_context_recall >= threshold else 'FAILED'} (threshold: {threshold})")

if all([avg_faithfulness >= threshold, avg_relevancy >= threshold, avg_context_recall >= threshold]):
    print("\nOVERALL: PASSED - System meets quality standards")
else:
    print("\nOVERALL: FAILED - System needs improvement")
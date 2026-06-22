import os
import json
import sqlite3
import time
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Helper function to print safely on Windows consoles
def safe_print(text: str):
    # Replace non-ASCII stars with ascii equivalent
    text = text.replace("★", "*")
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback to replacing unencodable characters
        print(text.encode("ascii", errors="replace").decode("ascii"))

# Helper function to extract JSON from LLM outputs that might contain text/markdown wrapping
def extract_json(text: str) -> dict:
    # Try direct parsing first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
        
    # Search for first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as je:
            raise ValueError(f"Found JSON-like block but it failed to parse: {json_str}. Error: {je}")
            
    raise ValueError(f"No JSON object found in text: {text}")

# Initialize DB to a known clean state before running evals
from setup_db import create_database, DB_PATH
create_database()

from shopping_agent import agent, llm

# Helper function to invoke agent with exponential backoff retries for rate limits
def invoke_agent_with_retry(payload: dict, max_retries: int = 5) -> dict:
    retry_delay = 20
    for attempt in range(max_retries):
        try:
            return agent.invoke(payload)
        except Exception as e:
            err_msg = str(e)
            if "rate_limit_exceeded" in err_msg or "429" in err_msg:
                safe_print(f"   [RATE LIMIT] Agent hit rate limit. Retrying in {retry_delay}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
                retry_delay = int(retry_delay * 1.5)
            else:
                raise e
    return agent.invoke(payload)

# Helper function to invoke LLM judge with exponential backoff retries for rate limits
def invoke_judge_with_retry(prompt: str, max_retries: int = 5) -> Any:
    retry_delay = 20
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            err_msg = str(e)
            if "rate_limit_exceeded" in err_msg or "429" in err_msg:
                safe_print(f"   [RATE LIMIT] Judge hit rate limit. Retrying in {retry_delay}s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(retry_delay)
                retry_delay = int(retry_delay * 1.5)
            else:
                raise e
    return llm.invoke(prompt)

# ==============================================================================
# TOOL CALL ACCURACY EVALUATION
# ==============================================================================

TOOL_ACCURACY_TEST_CASES = [
    {
        "query": "organic honey under $20",
        "expected_tool": "search_products",
        "expected_args": {
            "is_organic": True,
            "max_price": 20.0
        }
    },
    {
        "query": "what have I ordered before?",
        "expected_tool": "get_order_history",
        "expected_args": {}
    },
    {
        "query": "I always prefer organic products",
        "expected_tool": "save_user_preference",
        "expected_args": {
            "pref_key": "prefers_organic",
            "pref_value": "True"
        }
    },
    {
        "query": "get ratings for product 1",
        "expected_tool": "get_rating",
        "expected_args": {
            "product_id": 1
        }
    },
    {
        "query": "do you ship to alaska?",
        "expected_tool": "search_policy_and_faq",
        "expected_args": {}
    }
]


def run_tool_accuracy_eval():
    safe_print("======================================================================")
    safe_print("[INFO] RUNNING TOOL CALL ACCURACY EVALUATIONS")
    safe_print("======================================================================")
    
    passed = 0
    total = len(TOOL_ACCURACY_TEST_CASES)
    
    for i, tc in enumerate(TOOL_ACCURACY_TEST_CASES):
        query = tc["query"]
        expected_tool = tc["expected_tool"]
        expected_args = tc["expected_args"]
        
        safe_print(f"\nTest {i+1}: Query = \"{query}\"")
        if i > 0:
            time.sleep(10)
        
        try:
            result = invoke_agent_with_retry({"messages": [{"role": "user", "content": query}]})
            messages = result["messages"]
            
            tool_calls = []
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls.extend(msg.tool_calls)
            
            if not tool_calls:
                safe_print(f"[FAIL] No tool calls made. Expected call to '{expected_tool}'.")
                continue
                
            matched_tool = None
            for tc_call in tool_calls:
                if tc_call["name"] == expected_tool:
                    matched_tool = tc_call
                    break
                    
            if not matched_tool:
                actual_names = [tc_call["name"] for tc_call in tool_calls]
                safe_print(f"[FAIL] Did not call expected tool '{expected_tool}'. Called: {actual_names}")
                continue
                
            actual_args = matched_tool["args"]
            arg_mismatch = False
            for k, expected_val in expected_args.items():
                actual_val = actual_args.get(k)
                
                if isinstance(expected_val, float) and isinstance(actual_val, (int, float)):
                    if abs(expected_val - actual_val) > 1e-4:
                        arg_mismatch = True
                elif str(expected_val).strip().lower() != str(actual_val).strip().lower():
                    arg_mismatch = True
                    
            if arg_mismatch:
                safe_print(f"[FAIL] Tool args mismatch for '{expected_tool}'.")
                safe_print(f"   Expected: {expected_args}")
                safe_print(f"   Actual  : {actual_args}")
            else:
                safe_print(f"[PASS] Correctly called '{expected_tool}' with args {actual_args}")
                passed += 1
                
        except Exception as e:
            safe_print(f"[ERROR] Exception occurred during agent invoke: {e}")
            
    safe_print(f"\nTool Accuracy Summary: Passed {passed}/{total} tests.\n")
    return passed == total


# ==============================================================================
# RESPONSE QUALITY EVALUATION (LLM-AS-JUDGE)
# ==============================================================================

QUALITY_TEST_CASES = [
    {
        "query": "I want organic honey under $20 with 4.5+ rating",
        # Expected products: IDs 1, 5, 7
        "expected_products": "Product IDs: 1 (Organic Raw Honey), 5 (Organic Buckwheat Honey), 7 (Organic Acacia Honey)"
    }
]


def run_response_quality_eval():
    safe_print("======================================================================")
    safe_print("[INFO] RUNNING RESPONSE QUALITY EVALUATIONS (LLM-AS-JUDGE)")
    safe_print("======================================================================")
    
    passed = 0
    total = len(QUALITY_TEST_CASES)
    
    for i, tc in enumerate(QUALITY_TEST_CASES):
        query = tc["query"]
        expected_products = tc["expected_products"]
        
        safe_print(f"\nTest {i+1}: Evaluating response quality for: \"{query}\"")
        time.sleep(10)
        
        try:
            result = invoke_agent_with_retry({"messages": [{"role": "user", "content": query}]})
            assistant_response = result["messages"][-1].content
            
            safe_print(f"--- Assistant Response ---\n{assistant_response}\n--------------------------")
            
            judge_prompt = f"""You are an expert AI evaluator. Assess the quality of the shopping assistant response.

User query: {query}
Assistant Response: {assistant_response}
Expected Matching Products in database: {expected_products}

Score the assistant's response on a scale from 1 to 5 for each of these three criteria:
1. Relevance: Is the response on-topic, helpful, and directly addresses what the user asked?
2. Correctness: Did the assistant search and present the correct products? Check if it filtered out any product that is NOT organic, NOT honey, over $20, or under 4.5 rating.
3. Format Compliance: Did the assistant present the list in the EXACT plain text format:
   #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>
   with a blank line between each item? Bold, italic, backticks, or code blocks are NOT allowed.

You must reply with ONLY a valid JSON object matching the schema below. No introductory text or markdown formatting (like ```json).

Schema:
{{
  "relevance_score": <int 1-5>,
  "relevance_reason": "<string explanation>",
  "correctness_score": <int 1-5>,
  "correctness_reason": "<string explanation>",
  "format_score": <int 1-5>,
  "format_reason": "<string explanation>"
}}
"""
            time.sleep(10)
            
            judge_res = invoke_judge_with_retry(judge_prompt)
            raw_content = judge_res.content.strip()
            
            try:
                scores = extract_json(raw_content)
                
                safe_print(f"Relevance Score: {scores['relevance_score']}/5 | Reason: {scores['relevance_reason']}")
                safe_print(f"Correctness Score: {scores['correctness_score']}/5 | Reason: {scores['correctness_reason']}")
                safe_print(f"Format Score: {scores['format_score']}/5 | Reason: {scores['format_reason']}")
                
                if scores["relevance_score"] >= 4 and scores["correctness_score"] >= 4 and scores["format_score"] >= 4:
                    safe_print("[PASS] Response meets quality and format thresholds.")
                    passed += 1
                else:
                    safe_print("[FAIL] Response did not meet the quality thresholds (requires score >= 4 on all criteria).")
                    
            except ValueError as ve:
                safe_print(f"[ERROR] JSON extraction failed: {ve}")
                safe_print(f"[DEBUG] Raw judge output was:\n{raw_content}")
                
        except Exception as e:
            safe_print(f"[ERROR] Exception occurred during judge evaluation: {e}")
            
    safe_print(f"\nResponse Quality Summary: Passed {passed}/{total} tests.\n")
    return passed == total


if __name__ == "__main__":
    accuracy_ok = run_tool_accuracy_eval()
    quality_ok = run_response_quality_eval()
    
    if accuracy_ok and quality_ok:
        safe_print("[SUCCESS] ALL EVALUATIONS COMPLETED SUCCESSFULLY!")
        exit(0)
    else:
        safe_print("[WARN] SOME EVALUATIONS FAILED. Please review the output above.")
        exit(1)

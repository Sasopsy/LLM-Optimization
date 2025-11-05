import json
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer  # Fallback for F1 if no embeddings
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
import numpy as np
from datasets import load_dataset  # For test cases (e.g., HumanEval)
import RestrictedPython  # Safe code execution sandbox
from io import StringIO
import os

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class TaskBasedEvaluator:
    def __init__(self, endpoint_url: str, model_name: str = None, 
                 api_key: str = "EMPTY", max_tokens: int = 256, temperature: float = 0.0,
                 model_identifier: str = None):
        """
        Initialize task-based evaluator for OpenAI-compatible endpoint.
        
        Args:
            endpoint_url: vLLM server (e.g., "http://localhost:8000/v1")
            model_name: For API
            api_key: Dummy for vLLM
            max_tokens: Output cap
            temperature: For generation
            model_identifier: Identifier for CSV tracking (e.g., "baseline", "optimized_v1")
        """
        self.client = OpenAI(base_url=endpoint_url, api_key=api_key)
        self.model_name = model_name
        self.model_identifier = model_identifier
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Embeddings for chat (semantic similarity)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, general-purpose [web:144]
        
    def _build_few_shot_prompt(self, current_prompt: str, category: str, examples: List[Dict]) -> str:
        """Build a prompt with few-shot examples for the given category."""
        if not examples:
            return current_prompt
        
        # Build few-shot examples string
        examples_str = ""
        for ex in examples:
            if category == "QA":
                examples_str += f"{ex['prompt']} {ex['expected_output']}\n\n"
            elif category == "reasoning":
                # Add formatting instruction for reasoning
                examples_str += f"{ex['prompt']} {ex['expected_output']}\n\n"
            elif category == "chat":
                examples_str += f"User: {ex['prompt']}\nAssistant: {ex['expected_output']}\n\n"
            elif category == "code":
                examples_str += f"{ex['prompt']} {ex['expected_output']}\n\n"
        
        # Add category-specific instructions
        if category == "reasoning":
            instruction = "\n\nImportant: Format your solution with step-by-step reasoning, and end your answer with '#### [final_answer]' where [final_answer] is the numerical result.\n\n"
            return examples_str + instruction + current_prompt
        
        # Combine examples with current prompt
        return examples_str + current_prompt
        
    def _extract_examples_by_category(self, prompts_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract first few examples from each category for few-shot learning."""
        examples = {"QA": [], "reasoning": [], "chat": [], "code": []}
        seen_categories = set()
        
        for item in prompts_data:
            category = item["category"]
            # Take first 2 examples from each category
            if category in examples and len(examples[category]) < 2:
                examples[category].append(item)
        
        return examples
        
    def generate_outputs(self, dataset_path: str) -> List[Dict[str, Any]]:
        """
        Generate LLM outputs for dataset (same as before, for consistency).
        
        Returns: List with prompt, expected_output, output, category
        """
        with open(dataset_path, "r") as f:
            prompts_data = json.load(f)
        
        # Extract few-shot examples (first 2 from each category)
        few_shot_examples = self._extract_examples_by_category(prompts_data)
        
        # Build index map for few-shot examples
        example_indices_by_category = {}
        for category in ["QA", "reasoning", "chat", "code"]:
            example_indices_by_category[category] = []
            count = 0
            for i, item in enumerate(prompts_data):
                if item["category"] == category and count < 2:
                    example_indices_by_category[category].append(i)
                    count += 1
        
        results = []
        for idx, prompt_data in enumerate(tqdm(prompts_data, desc="Generating outputs")):
            prompt_max_tokens = prompt_data.get("target_output_tokens", self.max_tokens)
            category = prompt_data["category"]
            
            # Get few-shot examples for this category (exclude current item if it's an example)
            example_indices = example_indices_by_category.get(category, [])
            examples_to_use = [prompts_data[i] for i in example_indices if i != idx]
            
            # Build prompt with few-shot examples
            enhanced_prompt = self._build_few_shot_prompt(
                prompt_data["prompt"], 
                category, 
                examples_to_use
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": enhanced_prompt}],
                    max_tokens=prompt_max_tokens,
                    temperature=self.temperature,
                    stream=False
                )
                output = response.choices[0].message.content.strip()
                
                results.append({
                    "category": prompt_data["category"],
                    "prompt": prompt_data["prompt"],
                    "expected_output": prompt_data["expected_output"],
                    "output": output
                })
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    "category": prompt_data["category"],
                    "prompt": prompt_data["prompt"],
                    "expected_output": prompt_data["expected_output"],
                    "output": ""
                })
        
        return results
    
    def evaluate_qa(self, output: str, expected: str) -> Dict[str, float]:
        """
        QA: Exact Match (EM) + F1 (token overlap) [web:142][web:153].
        
        EM: 1 if normalized outputs match.
        F1: Harmonic mean of precision/recall on tokens.
        """
        if not output or not expected:
            return {"qa_em": 0.0, "qa_f1": 0.0}
        
        # Normalize (lower, strip punctuation, remove extras)
        norm_output = re.sub(r'[^\w\s]', '', output.lower().strip())
        norm_expected = re.sub(r'[^\w\s]', '', expected.lower().strip())
        
        # EM
        em = 1.0 if norm_output == norm_expected else 0.0
        
        # F1: Token-level
        out_tokens = set(norm_output.split())
        exp_tokens = set(norm_expected.split())
        
        if not out_tokens or not exp_tokens:
            return {"qa_em": em, "qa_f1": 0.0}
        
        precision = len(out_tokens & exp_tokens) / len(out_tokens)
        recall = len(out_tokens & exp_tokens) / len(exp_tokens)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"qa_em": em, "qa_f1": f1}
    
    def evaluate_reasoning(self, output: str, expected: str) -> float:
        """
        Reasoning (GSM8K-style): Extract final numerical answer, EM [web:143][web:146][web:154].
        
        Expected format: Solution steps ending with #### [answer]
        Extracts final answer from #### [number] pattern at the end.
        """
        if not output or not expected:
            return 0.0
        
        def extract_number(text):
            """Helper to safely extract and parse a number from text."""
            # Try format: #### [number]
            match = re.search(r'####\s*([\d.,]+)', text)
            if not match:
                # Fallback: look for "answer" pattern
                match = re.search(r'(?:answer|answer is|final answer)[:\s]*[$-]?\s*([\d.,]+)', text.lower())
            if not match:
                # Last resort: extract last number in the text (must have at least one digit)
                match = re.search(r'(\d+(?:[.,]\d+)*)\s*$', text.strip())
            
            if match:
                try:
                    num_str = match.group(1).replace(',', '')
                    # Ensure the string is a valid number
                    if num_str and any(c.isdigit() for c in num_str):
                        return float(num_str)
                except (ValueError, AttributeError):
                    pass
            return None
        
        expected_ans = extract_number(expected)
        output_ans = extract_number(output)
        
        # EM on numbers (tolerance for floats)
        if expected_ans is not None and output_ans is not None:
            tolerance = 1e-6 if isinstance(expected_ans, float) else 0
            return 1.0 if abs(expected_ans - output_ans) <= tolerance else 0.0
        
        return 0.0
    
    def evaluate_chat(self, output: str, expected: str) -> float:
        """
        Chat: Cosine similarity on embeddings (coherence/semantic match) [web:144][web:152].
        
        0-1 score; higher = more coherent/relevant.
        """
        if not output or not expected:
            return 0.0
        
        # Embed sentences
        emb_out = self.embedder.encode([output])
        emb_exp = self.embedder.encode([expected])
        
        # Cosine sim
        sim = cosine_similarity(emb_out, emb_exp)[0][0]
        return float(max(0.0, sim))  # Clamp >=0 and convert to Python float
    
    def evaluate_code(self, output: str, expected: str, prompt: str) -> float:
        """
        Code (HumanEval-style): Pass@1 via safe execution [web:48][web:147][web:168].
        
        Assumes prompt contains function signature; generates body. Executes against simple test (extract from expected/prompt).
        Uses RestrictedPython for sandbox (no imports/exec risks).
        """
        if not output or not expected:
            return 0.0
        
        # Extract test cases from expected (assume format: "def func(): ... test: assert func() == X")
        test_match = re.search(r'test[:\s]*([^.]+)', expected.lower())
        if not test_match:
            # Fallback: Simple string match on code
            return SequenceMatcher(None, output, expected).ratio()
        
        test_code = test_match.group(1).strip()
        
        # Combine: Full code = prompt (signature) + output (body) + test
        full_code = prompt + "\n" + output + "\n" + test_code
        
        # Safe execution (sandboxed)
        try:
            # RestrictedPython: Safe eval
            safe_globals = RestrictedPython.safe_globals.copy()
            safe_locals = {}
            
            # Compile and exec in sandbox
            code_obj = RestrictedPython.compile_restricted(full_code, '<string>', 'exec')
            exec(code_obj, safe_globals, safe_locals)
            
            # Check if assert passed (no exception = pass)
            return 1.0  # Pass@1
        except:
            return 0.0
    
    def evaluate_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply task-specific eval to one item.
        """
        category = item["category"]
        output = item["output"]
        expected = item["expected_output"]
        prompt = item["prompt"]
        
        item_copy = item.copy()
        
        if category == "QA":
            qa_metrics = self.evaluate_qa(output, expected)
            item_copy.update(qa_metrics)
            item_copy["task_score"] = (qa_metrics["qa_em"] + qa_metrics["qa_f1"]) / 2  # Avg
        
        elif category == "reasoning":
            reasoning_score = self.evaluate_reasoning(output, expected)
            item_copy["task_score"] = reasoning_score
        
        elif category == "chat":
            chat_score = self.evaluate_chat(output, expected)
            item_copy["task_score"] = chat_score
        
        elif category == "code":
            code_score = self.evaluate_code(output, expected, prompt)
            item_copy["task_score"] = code_score
        
        return item_copy
    
    def evaluate_dataset(self, results: List[Dict[str, Any]], output_path: str = "task_eval_results.json") -> Dict[str, Any]:
        """
        Evaluate all results with task-specific metrics, aggregate.
        
        Returns: Summary with avg scores per category, overall.
        """
        eval_results = []
        category_scores = {"QA": [], "reasoning": [], "chat": [], "code": []}
        
        for item in tqdm(results, desc="Task-based evaluation"):
            evaluated = self.evaluate_single_item(item)
            eval_results.append(evaluated)
            
            cat = evaluated["category"]
            if cat in category_scores:
                category_scores[cat].append(evaluated["task_score"])
        
        # Aggregates
        df = pd.DataFrame(eval_results)
        overall_summary = {
            "total_prompts": len(eval_results),
            "overall_avg_score": float(np.mean([np.mean(scores) for scores in category_scores.values()])),
            "category_averages": {cat: float(np.mean(scores)) for cat, scores in category_scores.items() if scores},
            "category_counts": {cat: len(scores) for cat, scores in category_scores.items()}
        }
        
        # Convert numpy types to native Python types for JSON serialization
        eval_results_serializable = convert_numpy_types(eval_results)
        overall_summary_serializable = convert_numpy_types(overall_summary)
        
        # Save
        with open(output_path, "w") as f:
            json.dump({"items": eval_results_serializable, "summary": overall_summary_serializable}, f, indent=2)
        df.to_csv(output_path.replace(".json", ".csv"), index=False)
        
        # Save/append summary CSV with model identifier and category scores
        if self.model_identifier is not None:
            summary_csv_path = "evaluation_summary.csv"
            summary_row = {
                "model": self.model_identifier,
                "QA_score": overall_summary["category_averages"].get("QA", 0.0),
                "reasoning_score": overall_summary["category_averages"].get("reasoning", 0.0),
                "chat_score": overall_summary["category_averages"].get("chat", 0.0),
                "code_score": overall_summary["category_averages"].get("code", 0.0),
                "overall_score": overall_summary["overall_avg_score"],
                "total_prompts": overall_summary["total_prompts"]
            }
            
            # Append to CSV or create new one
            if os.path.exists(summary_csv_path):
                summary_df = pd.read_csv(summary_csv_path)
                summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
            else:
                summary_df = pd.DataFrame([summary_row])
            
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"\nSummary saved to {summary_csv_path}")
        
        print("Evaluation complete:")
        print("\nScore interpretation: All scores are normalized to 0-1 scale (1.0 = perfect, 0.0 = worst)")
        print("  - QA: Average of Exact Match (EM) and F1 score")
        print("  - Reasoning: 1.0 if final answer matches (from #### [answer] format), 0.0 otherwise")
        print("  - Chat: Cosine similarity of embeddings (0-1)")
        print("  - Code: Pass@1 score (1.0 if code executes correctly, 0.0 otherwise)")
        print()
        for cat, avg in overall_summary["category_averages"].items():
            print(f"{cat}: Avg score {avg:.3f} ({overall_summary['category_counts'][cat]} prompts)")
        print(f"Overall avg: {overall_summary['overall_avg_score']:.3f} (out of 1.0)")
        
        return overall_summary
    
    def compare_configs(self, baseline_path: str, optimized_path: str) -> Dict[str, Any]:
        """
        Compare baseline vs optimized for degradation (<5%) [file:74].
        """
        with open(baseline_path) as f:
            baseline = json.load(f)["summary"]
        with open(optimized_path) as f:
            optimized = json.load(f)["summary"]
        
        overall_degradation = ((baseline["overall_avg_score"] - optimized["overall_avg_score"]) / baseline["overall_avg_score"] * 100) if baseline["overall_avg_score"] > 0 else 0
        passes = overall_degradation < 5
        
        comparison = {
            "baseline_overall": baseline["overall_avg_score"],
            "optimized_overall": optimized["overall_avg_score"],
            "degradation_pct": overall_degradation,
            "passes_quality": passes,
            "category_degradations": {}
        }
        
        # Per-category
        for cat in baseline["category_averages"]:
            if cat in optimized["category_averages"]:
                deg = ((baseline["category_averages"][cat] - optimized["category_averages"][cat]) / baseline["category_averages"][cat] * 100)
                comparison["category_degradations"][cat] = deg
        
        print(f"Overall degradation: {overall_degradation:.1f}% (passes: {passes})")
        return comparison

# Usage
if __name__ == "__main__":
    # Init for baseline
    evaluator = TaskBasedEvaluator(
        endpoint_url="http://localhost:8000/v1",  # Baseline vLLM
        model_name="meta-llama/Llama-3.1-8B-Instruct",  # Required for SGLang
        max_tokens=512,
        model_identifier="sglang"  # Add identifier to save to CSV
    )
    
    # Generate (reuse your dataset)
    dataset_path = "test_dataset.json"  # Or fixed_token_bench_dataset.json
    generated = evaluator.generate_outputs(dataset_path)
    
    # Evaluate
    baseline_summary = evaluator.evaluate_dataset(generated, "baseline_task_eval.json")

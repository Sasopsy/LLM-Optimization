from datasets import load_dataset
import json
from typing import List, Dict

def load_public_prompts(num_qa=200, num_reasoning=120, num_chat=120, num_code=80) -> List[Dict]:
    """Load diverse prompts from public datasets"""
    prompts = []
    
    # QA: SQuAD v2 (factual QA) - simpler structure than Natural Questions
    squad = load_dataset("squad_v2", split="train", streaming=True)
    qa_prompts = []
    for i, example in enumerate(squad):
        if i >= num_qa:
            break
        # SQuAD has simple structure: question and answers
        question_text = example['question']
        # Get the first answer text if available
        answer_text = ""
        if example['answers'] and example['answers']['text'] and len(example['answers']['text']) > 0:
            answer_text = example['answers']['text'][0]
        
        prompt = f"Question: {question_text}\nAnswer:"
        qa_prompts.append({
            "category": "QA",
            "prompt": prompt,
            "expected_output": answer_text
        })
    prompts.extend(qa_prompts)
    
    # Reasoning: GSM8K (math reasoning)
    gsm8k = load_dataset("gsm8k", "main", split="train", streaming=True)
    reasoning_prompts = []
    for i, example in enumerate(gsm8k):
        if i >= num_reasoning:
            break
        prompt = f"{example['question']}\nSolution:"
        reasoning_prompts.append({
            "category": "reasoning",
            "prompt": prompt,
            "expected_output": example['answer']
        })
    prompts.extend(reasoning_prompts)
    
    # Chat: Anthropic's hh-rlhf (conversational)
    hh = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
    chat_prompts = []
    for i, example in enumerate(hh):
        if i >= num_chat:
            break
        # Parse the conversation to extract human prompt and assistant response
        # hh-rlhf format: "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: ..."
        chosen_text = example['chosen']
        
        # Split by "\n\nAssistant:" to get turns
        parts = chosen_text.split('\n\nAssistant:')
        if len(parts) >= 2:
            # Get the first human turn as prompt (without 'Human:' tag)
            human_part = parts[0].replace('\n\nHuman:', '').strip()
            # Get the first assistant response (without 'Assistant:' tag)
            assistant_part = parts[1].split('\n\nHuman:')[0].strip()
            
            prompt = human_part
            chat_prompts.append({
                "category": "chat",
                "prompt": prompt,
                "expected_output": assistant_part
            })
    prompts.extend(chat_prompts)
    
    # Code: HumanEval (code generation)
    human_eval = load_dataset("openai_humaneval", split="test")
    code_prompts = []
    for i in range(min(num_code, len(human_eval))):
        example = human_eval[i]
        prompt = f"# {example['prompt']}\ndef solution({example['prompt'].split('def ')[1].split('(')[1]}):"
        code_prompts.append({
            "category": "code",
            "prompt": prompt,
            "expected_output": example['canonical_solution']
        })
    prompts.extend(code_prompts)
    
    # Save as JSON for reuse
    with open("test_dataset.json", "w") as f:
        json.dump(prompts, f, indent=2)
    
    return prompts

# Run this once
test_prompts = load_public_prompts()
print(f"Generated {len(test_prompts)} prompts: {len([p for p in test_prompts if p['category']=='QA'])} QA, etc.")

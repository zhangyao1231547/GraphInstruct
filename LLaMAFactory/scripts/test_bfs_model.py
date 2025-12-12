#!/usr/bin/env python3
"""
Test script for evaluating the fine-tuned Qwen3-4B BFS model.
Compares performance between base model and LoRA fine-tuned model.
"""

import json
import torch
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL_PATH = "/nvme0/work/workspaces-zy/model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507"
LORA_MODEL_PATH = "/mnt/yrfs/GraphInstruct/model/qwen3-4b-bfs-lora"
TEST_DATA_PATH = "/nvme0/work/workspaces-zy/GraphInstruct/LLaMAFactory/data/reasoning/BFS-int_id/train.json"

# Test cases (select a few from training data for validation)
TEST_INDICES = [100, 200, 500, 1000, 2000, 5000, 8000, 9500]

def load_test_cases(path, indices):
    """Load specific test cases from training data."""
    with open(path, 'r') as f:
        data = json.load(f)
    return [data[i] for i in indices if i < len(data)]

def extract_answer(text):
    """Extract the answer from model output."""
    # Look for <<<[...]>>> pattern
    match = re.search(r'<<<\[(.*?)\]>>>', text)
    if match:
        try:
            return [int(x.strip()) for x in match.group(1).split(',')]
        except:
            return None
    return None

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """Generate response from model."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def evaluate_model(model, tokenizer, test_cases, model_name="Model"):
    """Evaluate model on test cases."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    results = []
    correct = 0
    total = len(test_cases)

    for i, case in enumerate(test_cases):
        prompt = case['instruction'] + (f"\n{case['input']}" if case['input'] else "")
        expected = extract_answer(case['output'])

        start_time = time.time()
        response = generate_response(model, tokenizer, prompt)
        inference_time = time.time() - start_time

        predicted = extract_answer(response)
        is_correct = predicted == expected
        if is_correct:
            correct += 1

        result = {
            "test_id": i + 1,
            "original_id": case.get('id', 'N/A'),
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "inference_time": round(inference_time, 2),
            "response": response[:500] + "..." if len(response) > 500 else response
        }
        results.append(result)

        status = "PASS" if is_correct else "FAIL"
        print(f"Test {i+1}/{total}: {status} (Time: {inference_time:.2f}s)")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted}")
        print()

    accuracy = correct / total * 100 if total > 0 else 0
    print(f"\n{model_name} Results:")
    print(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print(f"  Avg Inference Time: {sum(r['inference_time'] for r in results)/len(results):.2f}s")

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results
    }

def main():
    print("="*60)
    print("BFS Model Evaluation Script")
    print("="*60)

    # Load test cases
    print("\nLoading test cases...")
    test_cases = load_test_cases(TEST_DATA_PATH, TEST_INDICES)
    print(f"Loaded {len(test_cases)} test cases")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    # Test 1: Fine-tuned LoRA model
    print("\nLoading fine-tuned LoRA model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    lora_model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    lora_model.eval()

    lora_results = evaluate_model(lora_model, tokenizer, test_cases, "Qwen3-4B + BFS-LoRA")

    # Clean up LoRA model
    del lora_model
    del base_model
    torch.cuda.empty_cache()

    # Test 2: Base model (for comparison)
    print("\nLoading base model for comparison...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    base_model.eval()

    base_results = evaluate_model(base_model, tokenizer, test_cases, "Qwen3-4B (Base)")

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\n{'Model':<30} {'Accuracy':<15} {'Correct':<10}")
    print("-"*55)
    print(f"{'Qwen3-4B (Base)':<30} {base_results['accuracy']:.1f}%{'':<9} {base_results['correct']}/{base_results['total']}")
    print(f"{'Qwen3-4B + BFS-LoRA':<30} {lora_results['accuracy']:.1f}%{'':<9} {lora_results['correct']}/{lora_results['total']}")

    improvement = lora_results['accuracy'] - base_results['accuracy']
    print(f"\nImprovement: {improvement:+.1f}%")

    # Save results to file
    all_results = {
        "base_model": base_results,
        "lora_model": lora_results,
        "improvement": improvement
    }

    output_path = "/mnt/yrfs/GraphInstruct/model/qwen3-4b-bfs-lora/evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return all_results

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test reward functions with different prompt variants."""

import re
import torch

device = 'cpu'

# Copy the reward functions
def accuracy_reward_fn(texts, eval_fn):
    """Check if the model's answer is correct."""
    accuracy_rewards = []
    for text in texts:
        # Try to extract answer from <answer> tags
        answers = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
        if answers:
            # Use the last answer tag if multiple exist
            reward = 1.0 if eval_fn(answers[-1].strip()) else 0.0
        else:
            # For 'free' mode or when no tags, try to extract the last line or number
            lines = text.strip().split('\n')
            last_line = lines[-1].strip() if lines else ""
            reward = 1.0 if eval_fn(last_line) else 0.0
        accuracy_rewards.append(reward)
    return torch.tensor(accuracy_rewards, device=device)

def format_reward_fn(texts, variant='original'):
    """Check if the model follows the requested format."""
    format_rewards = []
    
    for text in texts:
        # Split to get only the assistant's response
        parts = text.split('assistant')
        if len(parts) < 2:
            format_rewards.append(0.0)
            continue
        assistant_response = parts[-1].strip()
        
        if variant == 'no_thinking':
            # Only require <answer> tag, no thinking
            has_answer = bool(re.search(r'<answer>.*?</answer>', assistant_response, re.DOTALL))
            has_think = bool(re.search(r'<think>', assistant_response))
            reward = 1.0 if (has_answer and not has_think) else 0.0
        
        elif variant in ['original', 'strict', 'simple']:
            # Require both <think> and <answer> tags
            has_both = bool(re.search(r'<think>.*?</think>\s*<answer>.*?</answer>', assistant_response, re.DOTALL))
            reward = 1.0 if has_both else 0.0
        
        elif variant == 'free':
            # Free mode: give reward if answer is provided
            has_answer_tag = bool(re.search(r'<answer>.*?</answer>', assistant_response, re.DOTALL))
            has_content = len(assistant_response.strip()) > 10
            reward = 1.0 if (has_answer_tag or has_content) else 0.0
        
        else:
            # Default: require both tags
            has_both = bool(re.search(r'<think>.*?</think>\s*<answer>.*?</answer>', assistant_response, re.DOTALL))
            reward = 1.0 if has_both else 0.0
        
        format_rewards.append(reward)
    
    return torch.tensor(format_rewards, device=device)

# Test cases
def test_variant(variant, system_prompt, response, expected_format_reward):
    """Test a specific variant."""
    full_text = f"system\n{system_prompt}\nuser\nSolve for x: 2x + 5 = 15\nassistant\n{response}"
    
    # Simple eval function that checks if answer is "5"
    def eval_fn(ans):
        return ans.strip() == "5"
    
    acc_reward = accuracy_reward_fn([full_text], eval_fn)
    fmt_reward = format_reward_fn([full_text], variant=variant)
    
    print(f"\n{'='*60}")
    print(f"Variant: {variant}")
    print(f"System: {system_prompt[:50]}...")
    print(f"Response: {response[:100]}...")
    print(f"Accuracy Reward: {acc_reward.item():.1f} (expected: 1.0)")
    print(f"Format Reward: {fmt_reward.item():.1f} (expected: {expected_format_reward})")
    
    if fmt_reward.item() == expected_format_reward:
        print("✅ PASS")
    else:
        print("❌ FAIL")
    
    return acc_reward.item(), fmt_reward.item()

print("Testing Reward Functions")
print("="*60)

# Test 'original' variant
test_variant(
    'original',
    "Respond in the following format: <think> ... </think> <answer> ... </answer>",
    "<think>2x + 5 = 15, so 2x = 10, so x = 5</think> <answer>5</answer>",
    1.0
)

# Test 'strict' variant with correct format
test_variant(
    'strict',
    "You must respond in exactly this format: <think> ... </think> <answer> ... </answer>",
    "<think>Step 1: 2x + 5 = 15\nStep 2: 2x = 10\nStep 3: x = 5</think> <answer>5</answer>",
    1.0
)

# Test 'strict' variant with incorrect format (missing think)
test_variant(
    'strict',
    "You must respond in exactly this format: <think> ... </think> <answer> ... </answer>",
    "<answer>5</answer>",
    0.0
)

# Test 'no_thinking' variant with correct format
test_variant(
    'no_thinking',
    "Provide your answer directly in the following format: <answer> ... </answer>",
    "<answer>5</answer>",
    1.0
)

# Test 'no_thinking' variant with incorrect format (has thinking)
test_variant(
    'no_thinking',
    "Provide your answer directly in the following format: <answer> ... </answer>",
    "<think>2x = 10</think> <answer>5</answer>",
    0.0
)

# Test 'free' variant with tags
test_variant(
    'free',
    "Solve the problem. You may use <think> and <answer> tags if helpful.",
    "<think>2x + 5 = 15</think> <answer>5</answer>",
    1.0
)

# Test 'free' variant without tags (just text)
test_variant(
    'free',
    "Solve the problem. You may use <think> and <answer> tags if helpful.",
    "To solve 2x + 5 = 15, we subtract 5 from both sides to get 2x = 10, then divide by 2 to get x = 5.",
    1.0  # Should still get format reward for providing content
)

# Test 'simple' variant
test_variant(
    'simple',
    "Think step-by-step, then provide your answer: <think> ... </think> <answer> ... </answer>",
    "<think>Subtract 5: 2x = 10\nDivide by 2: x = 5</think> <answer>5</answer>",
    1.0
)

print("\n" + "="*60)
print("Testing complete!")

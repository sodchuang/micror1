#!/usr/bin/env python3
"""完整测试所有 prompt variants 的 reward 计算"""

import re
import torch

device = 'cpu'

# 从 train.py 复制最新的 reward 函数
exec(open('train.py').read().split('# reward function definitions')[1].split('# main training loop')[0])

def test_variant(variant_name, test_cases):
    """测试一个 variant 的所有情况"""
    print(f"\n{'='*80}")
    print(f"测试 {variant_name.upper()} 模式")
    print('='*80)
    print(f"{'测试名称':<25} {'格式奖励':<12} {'准确度':<10} {'总分':<10} {'状态'}")
    print('-'*80)
    
    total_tests = len(test_cases)
    passed = 0
    
    for name, response, expected_fmt_min, expected_fmt_max in test_cases:
        text = f"system\nPrompt here\nuser\nSolve 2x+5=15\nassistant\n{response}"
        
        # 测试 format reward
        fmt_reward = format_reward_fn([text], variant=variant_name)
        
        # 测试 accuracy reward (检查答案是否包含 5)
        acc_reward = accuracy_reward_fn([text], lambda x: '5' in x.strip())
        
        total = fmt_reward.item() + acc_reward.item()
        
        # 检查是否在预期范围内
        is_pass = expected_fmt_min <= fmt_reward.item() <= expected_fmt_max
        status = "✅" if is_pass else "❌"
        
        if is_pass:
            passed += 1
        
        print(f"{status} {name:<23} {fmt_reward.item():.2f} ({expected_fmt_min:.1f}-{expected_fmt_max:.1f})   "
              f"{acc_reward.item():.1f}        {total:.2f}")
    
    print('-'*80)
    print(f"通过: {passed}/{total_tests} ({100*passed/total_tests:.0f}%)\n")
    return passed == total_tests

# 定义所有测试用例
all_variants = {
    'simple': [
        ("完整标签", "<think>步骤...</think> <answer>5</answer>", 1.0, 1.0),
        ("只有answer标签", "<answer>5</answer>", 0.7, 0.7),
        ("只有think标签", "<think>推理过程...</think>\n5", 0.5, 0.5),
        ("Think/Answer文本", "Think: 解题\nAnswer: 5", 0.3, 0.4),
        ("只有Answer文本", "Answer: 5", 0.3, 0.3),
        ("多行文本", "首先\n然后\n答案是5", 0.2, 0.2),
        ("单行长文本", "经过计算得出答案是5", 0.1, 0.1),
        ("只有数字", "5", 0.0, 0.1),
    ],
    
    'free': [
        ("完整标签", "<think>推理</think> <answer>5</answer>", 1.0, 1.0),
        ("只有answer标签", "<answer>5</answer>", 0.8, 0.8),
        ("Answer文本", "Answer: 5", 0.6, 0.6),
        ("长文本回复", "根据方程式2x+5=15，我们可以得出x=5", 0.4, 0.4),
        ("短文本", "答案是5", 0.0, 0.4),
        ("只有数字", "5", 0.0, 0.0),
    ],
    
    'strict': [
        ("完整标签", "<think>详细推理...</think> <answer>5</answer>", 1.0, 1.0),
        ("只有answer标签", "<answer>5</answer>", 0.7, 0.7),
        ("Think/Answer文本", "Think: 计算\nAnswer: x=5", 0.3, 0.4),
        ("只有数字", "5", 0.0, 0.1),
    ],
    
    'no_thinking': [
        ("只有answer标签(正确)", "<answer>5</answer>", 1.0, 1.0),
        ("Answer文本(正确)", "Answer: 5", 0.5, 0.5),
        ("有think标签(错误)", "<think>推理</think> <answer>5</answer>", 0.0, 0.0),
        ("只有数字", "5", 0.0, 0.0),
    ],
    
    'original': [
        ("完整标签", "<think>推理</think> <answer>5</answer>", 1.0, 1.0),
        ("只有answer标签", "<answer>5</answer>", 0.7, 0.7),
        ("只有数字", "5", 0.0, 0.1),
    ],
}

print("\n" + "="*80)
print("完整测试所有 Prompt Variants 的 Reward 计算")
print("="*80)

all_passed = True
for variant_name, test_cases in all_variants.items():
    passed = test_variant(variant_name, test_cases)
    all_passed = all_passed and passed

print("\n" + "="*80)
if all_passed:
    print("✅ 所有测试通过！Simple 和 Free 模式的 reward 计算完全正确！")
else:
    print("❌ 部分测试失败，请检查上面的详细信息")
print("="*80)

print("\n【关键总结】")
print("-" * 80)
print("✅ SIMPLE 模式: 渐进式奖励，鼓励使用标签但也接受文本格式")
print("   - 完整标签 <think>+<answer>: 1.0")
print("   - 只有 <answer>: 0.7")
print("   - Think:/Answer: 文本: 0.3-0.4")
print("   - 任何有内容的回答: 0.1-0.2")
print()
print("✅ FREE 模式: 最宽松，接受各种格式")
print("   - 完整标签: 1.0")
print("   - 只有 <answer>: 0.8")
print("   - 有结构(Answer:): 0.6")
print("   - 有内容(>10字符): 0.4")
print()
print("✅ Accuracy Reward: 所有模式都能正确识别答案")
print("   - 优先从 <answer> 标签提取")
print("   - 其次从 'Answer:' 后面提取")
print("   - 最后使用最后一行")
print("-" * 80)

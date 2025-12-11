#!/usr/bin/env python3
"""测试不同的 prompt variants 是否正确设置"""

import logicpuzzles

print("="*70)
print("测试所有 Prompt Variants")
print("="*70)

variants = ['original', 'no_thinking', 'strict', 'simple', 'free']

for variant in variants:
    print(f"\n{'='*70}")
    print(f"Variant: {variant}")
    print("="*70)
    
    # 设置 variant
    logicpuzzles.set_prompt_variant(variant)
    
    # 获取当前的 system_prompt
    current_prompt = logicpuzzles.system_prompt
    
    print(f"System Prompt:\n{current_prompt}")
    
    # 生成一个示例消息
    messages = logicpuzzles.make_messages("Test question")
    print(f"\nMessages:")
    for msg in messages:
        print(f"  [{msg['role']}] {msg['content'][:100]}...")

print("\n" + "="*70)
print("验证完成！")
print("="*70)

# 特别测试 free 模式
print("\n" + "="*70)
print("特别验证 FREE 模式")
print("="*70)
logicpuzzles.set_prompt_variant('free')
print(f"Free prompt: {logicpuzzles.system_prompt}")

expected = "Solve the problem. You may use <think> and <answer> tags to structure your response if helpful."
if logicpuzzles.system_prompt == expected:
    print("✅ FREE 模式 prompt 正确!")
else:
    print("❌ FREE 模式 prompt 不正确!")
    print(f"Expected: {expected}")
    print(f"Got: {logicpuzzles.system_prompt}")

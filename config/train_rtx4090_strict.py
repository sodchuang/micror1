# A configuration sized for training Qwen2.5-1.5B-Instruct on an RTX 4090 (24Gb mem).
# Tested on a 2x4090 node.
#
# launch as the following (e.g. in a screen session) and wait ~1 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_rtx4090.py

wandb_log = True
wandb_run_name='r1-qwen2p5-1p5B-instruct-strict'

checkpoint_path = 'Qwen/Qwen2.5-1.5B-Instruct'


import logicpuzzles

prompt_variant = 'strict'      # 'original', 'no_thinking', 'strict', 'simple', 'free'
simplified_prompts = False

# --- REDUCED VRAM SETTINGS ---
device_rollout_batch_size = 32
group_size = 8
episodes_per_rollout = 64 # Changed from 512

policy_epochs = 1

max_new_tokens = 512

policy_update_batch_size = 2
# -----------------------------

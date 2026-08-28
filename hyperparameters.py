# Blacksmith-1A: ~522M params for Radeon 8060S iGPU (~15 TFLOP/s effective bf16)
# d=1280, 10 heads (d_head=128), 20 layers
# VRAM at batch 32: ~21GB (6.3 weights/opt + ~15 activations) - lots of headroom
#
# Token budget: 10B tokens (~19 tok/param, Chinchilla-ish) = 76k steps
# at ~36s/step on the iGPU = ~32 days. Resume-safe: checkpoints every 50 steps.
#
# The full 42B-token corpus (80 tok/param = 134 days) is possible but only
# worth it if loss is still falling steeply at 76k steps - decide then.

d_model = 1280
n_heads = 10
n_layers = 20
max_seq_len = 4096
batch_size = 32
n_steps = 5000
total_steps = 76000
lr_peak = 3e-4
warmup_steps = 1500
weight_decay = 1e-2

# # Blacksmith-1A: 1.62B params (needs a real GPU - ~6 min/step on 8060S)
# d_model = 2048
# n_heads = 16
# n_layers = 28
# max_seq_len = 4096
# batch_size = 64
# n_steps = 5000
# total_steps = 200000
# lr_peak = 2.5e-4
# warmup_steps = 4000
# weight_decay = 1e-2

# # Blacksmith-1: GPT-2 Small (162M)
# d_model = 768
# n_heads = 12
# n_layers = 12
# max_seq_len = 1024
# batch_size = 32
# n_steps = 5000
# total_steps = 100000
# lr_peak = 2.5e-4
# warmup_steps = 2000
# weight_decay = 1e-2

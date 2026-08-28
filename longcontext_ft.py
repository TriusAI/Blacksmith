import os
import glob
import pathlib
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import NanoGPT
from tokenmaker import Tokenizer, TokenizerVocabSize
from dataset_mixed import MixedDatasetManager
from hyperparameters import d_model, n_heads, n_layers

# Long-context fine-tuning: extend the 4096-trained model through
# 8192 -> 16384 -> 32768 -> 65536, then extrapolate to 131072 with YaRN
# at inference.
#
#   torchrun --nproc_per_node=N --master_addr=127.0.0.1 --master_port=29500 \
#       ./longcontext_ft.py --phase=0
#
# Run phases in order 0, 1, 2, 3. Each phase loads the previous phase's
# checkpoint and trains with rope_scale=1 (the context extension here is
# achieved by TRAINING at longer positions, not by scaling).
#
# Sized for the 522M model (d=1280, l=20) on Radeon 8060S: each phase
# fits in ~21-27GB and runs ~72s/step; the full pipeline is ~5.5 days.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=int, required=True)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Phase configurations. Each phase trains at a longer max_seq_len than the
# last. batch sizes halve as seq doubles to keep stored activations flat.
# Steps shrink per phase: the model adapts to longer positions quickly once
# it has seen the previous length.
# ---------------------------------------------------------------------------
PHASES = [
    # (seq, batch, steps, lr_peak, warmup, in_ckpt, out_ckpt)
    (8192,  32, 2000, 2.5e-5, 100, 'blacksmith-2_model.pt', 'blacksmith-2_lc8k.pt'),
    (16384, 16, 2000, 1.2e-5, 100, 'blacksmith-2_lc8k.pt',  'blacksmith-2_lc16k.pt'),
    (32768, 8,  1500, 6e-6,   100, 'blacksmith-2_lc16k.pt', 'blacksmith-2_lc32k.pt'),
    (65536, 4,  1000, 3e-6,   100, 'blacksmith-2_lc32k.pt', 'blacksmith-2_lc64k.pt'),
]

seq_len, batch_size, n_steps, lr_peak, warmup_steps, in_ckpt, out_ckpt = PHASES[args.phase]
total_steps = n_steps
weight_decay = 1e-2

# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
acc = torch.accelerator.current_accelerator()
backend = torch.distributed.get_default_backend_for_device(acc)
dist.init_process_group(backend)

rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
device = f'cuda:{local_rank}'
is_main = (rank == 0)

if is_main:
    print(f'phase={args.phase} seq={seq_len} batch={batch_size} '
          f'steps={n_steps} lr={lr_peak}')

# During long-context FT, rope is NOT scaled (rope_scale=1): we are TRAINING
# the model at longer positions directly. YaRN scaling is only applied at
# inference to stretch past the final trained length.
model = NanoGPT(TokenizerVocabSize, d_model, n_heads, n_layers, seq_len,
                rope_scale=1.0, original_max_pos=4096).to(device)
model = model.to(torch.bfloat16)

if not pathlib.Path(in_ckpt).exists():
    raise FileNotFoundError(f'input checkpoint not found: {in_ckpt}')
chkp = torch.load(in_ckpt, map_location=device)
sd = {k.replace('module.', '', 1): v for k, v in chkp['model'].items()}
model.load_state_dict(sd)
if is_main:
    print(f'loaded {in_ckpt} (step {chkp.get("step", "?")})')

optimizer = optim.AdamW(model.parameters(), lr=lr_peak, weight_decay=weight_decay)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()) * (1 - 0.01) + 0.01

last_step = 0
if pathlib.Path(out_ckpt).exists() and is_main:
    print(f'{out_ckpt} exists — delete it to restart this phase.')
for pg in optimizer.param_groups:
    pg.setdefault('initial_lr', lr_peak)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

model = DDP(model, device_ids=[local_rank])

# ---------------------------------------------------------------------------
# Data: same mixed corpus, sampled at the phase's longer window.
# ---------------------------------------------------------------------------
dataset = MixedDatasetManager(device, [
    ('slimpajama', sorted(glob.glob('slimpajama_subset_*.bin')), 0.40),
    ('cosmopedia', sorted(glob.glob('cosmopediav2_*.bin')),      0.35),
    ('ultrachat',  ['UltraChat_uint16le.bin'],                    0.15),
    ('openthought', sorted(glob.glob('openthought_*.bin')),       0.10),
])
if is_main:
    print('training data loaded')

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
if is_main:
    print('long-context fine-tuning started')

loss_counting = []
for step in range(n_steps):
    inputs, targets = dataset.get_batch(batch_size, seq_len)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss, _, _ = model(inputs, targets=targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if len(loss_counting) >= 50:
        loss_counting = loss_counting[1:]
    loss_counting.append(loss.item())

    if is_main:
        avg = sum(loss_counting) / len(loss_counting)
        print(f'Step {step}: loss={loss.item():.4f} avgloss={avg:.4f}')
    if step % 50 == 0 and is_main:
        torch.save({
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
        }, out_ckpt)
    dist.barrier()

if is_main:
    torch.save({
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': n_steps,
    }, out_ckpt)
    print(f'phase complete: {out_ckpt}')

dist.barrier()
dist.destroy_process_group()
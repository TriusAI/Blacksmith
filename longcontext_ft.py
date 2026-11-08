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
# 8192 -> 16384 -> 32768 -> 65536, then extrapolate past 64k with YaRN at
# inference.
#
#   torchrun --nproc_per_node=N --master_addr=127.0.0.1 --master_port=29500 \
#       ./longcontext_ft.py --phase=0
#
# Run phases in order 0, 1, 2, 3. Each phase loads the previous phase's
# checkpoint and trains with rope_scale=1 (the context extension here is
# achieved by TRAINING at longer positions, not by scaling). Each phase
# checkpoints every 100 steps and RESUMES from its own output checkpoint,
# so a crash or a pause only costs the steps since the last save.
#
# 65536 is a deliberately short phase: attention FLOPs at 64k are ~2x the FFN
# FLOPs, so a step costs roughly twice the 32k step, and the 8k/16k/32k phases
# all stopped improving well inside their budgets. 600 steps is the same token
# budget per length step that the earlier phases actually used before they
# flattened. Beyond the trained length, stretch with YaRN at inference:
#   python3 generate.py --ckpt blacksmith-1_lc64k.pt --rope-scale 2

import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=int, required=True)
# Overrides, mainly for smoke tests and VRAM tuning. They do not change the
# phase table, and --out keeps scratch runs from polluting the real names.
parser.add_argument('--steps', type=int, default=None,
                    help='override the phase step count')
parser.add_argument('--batch-size', type=int, default=None,
                    help='override the phase batch size')
parser.add_argument('--out', type=str, default=None,
                    help='override the phase output checkpoint path')
parser.add_argument('--no-compile', action='store_true',
                    help='skip torch.compile (slower, but no compile warmup)')
parser.add_argument('--save-every', type=int, default=100,
                    help='checkpoint interval in steps. At 64k a step takes '
                         '~7 min, so 100 steps is nearly 12 hours of exposure; '
                         'drop it for the long phases')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Phase configurations. Each phase trains at a longer max_seq_len than the
# last. batch sizes halve as seq doubles to keep VRAM roughly flat: the
# stored per-block activations scale with batch*seq, so 4k*32 and 8k*32
# differ mainly by attention cost, not by activation bulk.
# ---------------------------------------------------------------------------
PHASES = [
    # (seq, batch, steps, lr_peak, warmup, in_ckpt, out_ckpt)
    (8192,  32, 3000, 2.5e-5, 100, 'blacksmith-1_model.pt',   'blacksmith-1_lc8k.pt'),
    (16384, 16, 2000, 1.2e-5, 100, 'blacksmith-1_lc8k.pt',    'blacksmith-1_lc16k.pt'),
    (32768, 8,  1500, 6e-6,   100, 'blacksmith-1_lc16k.pt',   'blacksmith-1_lc32k.pt'),
    (65536, 4,  600,  3e-6,   100, 'blacksmith-1_lc32k.pt',   'blacksmith-1_lc64k.pt'),
]

seq_len, batch_size, n_steps, lr_peak, warmup_steps, in_ckpt, out_ckpt = PHASES[args.phase]
if args.batch_size is not None:
    batch_size = args.batch_size
if args.steps is not None:
    n_steps = args.steps
if args.out is not None:
    out_ckpt = args.out
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

# Resume from this phase's own checkpoint if one exists (a crash or a pause
# costs at most the steps since the last save every 100 steps). A fresh phase
# deletes out_ckpt to start over.
last_step = -1
if pathlib.Path(out_ckpt).exists():
    rk = torch.load(out_ckpt, map_location=device)
    model.load_state_dict(
        {k.replace('module.', '', 1): v for k, v in rk['model'].items()})
    optimizer.load_state_dict(rk['optimizer'])
    last_step = int(rk.get('step', -1))
    if is_main:
        print(f'resuming {out_ckpt} at step {last_step + 1}')

start_step = last_step + 1
for pg in optimizer.param_groups:
    pg.setdefault('initial_lr', lr_peak)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda,
                                        last_epoch=start_step - 1)

# DDP only pays off with more than one rank; with a single process it just
# adds a gradient-bucket copy, so mirror main.py and skip it.
if world_size > 1:
    model = DDP(model, device_ids=[local_rank])
raw_model = model.module if world_size > 1 else model

# torch.compile matters here: long-context steps are compute-bound and the
# eager path is measurably slower (main.py has always used it).
if not args.no_compile:
    if is_main:
        print('compiling model...')
    model = torch.compile(model)
    if is_main:
        print('model compiled.')

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

# Stop cleanly on SIGTERM/SIGINT: finish the current step, checkpoint, exit.
# Without this, pausing a run keeps its VRAM and killing it throws away every
# step since the last periodic (100-step) save.
#
# IMPORTANT: send the signal to the WORKER pid (the `python3 -u
# ./longcontext_ft.py` child of torchrun), not to the torchrun pid. torchrun's
# elastic agent gives children only a few seconds to die before SIGKILLing
# them, which is shorter than a long-context step (77s at 8k), so signalling
# torchrun loses the in-flight step and the checkpoint write:
#     kill -TERM $(pgrep -P <torchrun pid>)
import signal
stop_requested = False
def _request_stop(signum, _frame):
    global stop_requested
    stop_requested = True
    if is_main:
        print(f'got signal {signum}: stopping after this step', flush=True)
for _sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(_sig, _request_stop)

loss_counting = []
step_t0 = time.perf_counter()
last_completed = start_step - 1
for step in range(start_step, n_steps):
    inputs, targets = dataset.get_batch(batch_size, seq_len)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        loss, _, _ = model(inputs, targets=targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    last_completed = step

    if len(loss_counting) >= 50:
        loss_counting = loss_counting[1:]
    loss_counting.append(loss.item())

    if is_main:
        t_now = time.perf_counter()
        avg = sum(loss_counting) / len(loss_counting)
        print(f'Step {step}: loss={loss.item():.4f} avgloss={avg:.4f} '
              f'step_time={t_now - step_t0:.1f}s')
        step_t0 = t_now
    if step % args.save_every == 0 and is_main:
        # Save the unwrapped module's state_dict (no "module." prefix).
        torch.save({
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
        }, out_ckpt)
    # All ranks stay in sync at the save boundary.
    dist.barrier()

    if stop_requested:
        if is_main:
            print(f'stop requested: checkpointing at step {step} and exiting')
        break

if is_main and last_completed >= 0:
    torch.save({
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': last_completed,
    }, out_ckpt)
    if last_completed < n_steps - 1:
        print(f'checkpoint written: {out_ckpt} at step {last_completed}')
    else:
        print(f'phase complete: {out_ckpt}')

dist.barrier()
dist.destroy_process_group()
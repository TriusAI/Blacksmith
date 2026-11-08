import os
import pathlib
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from model import NanoGPT
from tokenmaker import Tokenizer, TokenizerVocabSize
from sft_dataset import SFTDataset, PackedSFTDataset
from hyperparameters import d_model, n_heads, n_layers, max_seq_len

# torchrun --nproc_per_node=N --master_addr=127.0.0.1 --master_port=29500 ./sft.py
#
# Supervised fine-tuning. The dataset pads to --data-seq-len while the model
# keeps RoPE tables out to --model-seq-len, so padding can be shortened for
# speed and the model length raised after long-context training without
# touching the architecture.

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=None,
                    help='override n_steps (pilot runs)')
parser.add_argument('--lr', type=float, default=None,
                    help='override lr_peak; use ~2e-5 when continuing from an '
                         'already fine-tuned checkpoint so chat behaviour is '
                         'not overwritten')
parser.add_argument('--warmup', type=int, default=None,
                    help='override warmup_steps; pilots shorter than 500 '
                         'steps otherwise spend their whole run warming up')
parser.add_argument('--batch-size', type=int, default=None)
parser.add_argument('--data-seq-len', type=int, default=2048,
                    help='padding/truncation length for SFT samples. '
                         'UltraChat conversations measure median 1124 tokens, '
                         'p95 2042, max 2875, so 2048 covers ~95%% while '
                         'halving the padding compute of 4096')
parser.add_argument('--model-seq-len', type=int, default=max_seq_len,
                    help='RoPE table length. Raise to the long-context '
                         'trained length when SFTing an LC checkpoint.')
parser.add_argument('--pretrained', type=str, default='blacksmith-1_model.pt')
parser.add_argument('--out', type=str, default='blacksmith-1_sft.pt')
parser.add_argument('--pack-len', type=int, default=0,
                    help='>0 concatenates conversations into fixed windows of '
                         'this length instead of padding each one (LongAlign '
                         'packing). Keeps the model seeing long positions; use '
                         'with a smaller --batch-size since memory scales with '
                         'pack_len * batch_size')
parser.add_argument('--max-tokens', type=int, default=None,
                    help='cap the packed stream at this many tokens')
parser.add_argument('--data', type=str, default=None,
                    help='comma separated path:repeat list, overriding data_files. '
                         'e.g. "identity.jsonl:10,train_0.jsonl:1,longalign_sft.jsonl:1"')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# SFT hyperparameters
# ---------------------------------------------------------------------------
batch_size = 16
n_steps = 10000
lr_peak = 5e-5
warmup_steps = 500
weight_decay = 1e-2
ckpt_path = args.out
pretrained_path = args.pretrained
data_seq_len = args.data_seq_len
model_seq_len = args.model_seq_len
if args.batch_size is not None:
    batch_size = args.batch_size
if args.steps is not None:
    n_steps = args.steps
if args.warmup is not None:
    warmup_steps = args.warmup
if args.lr is not None:
    lr_peak = args.lr
total_steps = n_steps

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
    print(f'world_size={world_size} device={device} backend={backend}')

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# Add or remove JSONL files here. Each file has one conversation per line,
# either {"data": ["msg1", "msg2", ...]} (UltraChat format) or
# {"messages": [...]} format.
if args.data:
    data_files = []
    for item in args.data.split(','):
        path, _, rep = item.strip().partition(':')
        data_files.append((path, int(rep) if rep else 1))
else:
    data_files = [
        # 82 identity examples. 200x made them ~10% of every batch, which leaked
        # "I'm Blacksmith, trained by Trius" into unrelated answers (it showed
        # up as the reply to "what is 2+2"). 25x keeps the identity present
        # without making it the dominant response prior.
        ('identity.jsonl', 25),
        ('train_0.jsonl', 1),
    ]
if args.pack_len > 0:
    dataset = PackedSFTDataset(data_files, pack_len=args.pack_len,
                               max_tokens=args.max_tokens, verbose=is_main)
    data_seq_len = args.pack_len
else:
    dataset = SFTDataset(data_files, max_seq_len=data_seq_len)
if is_main:
    print(f'SFT samples: {len(dataset)} (data_seq_len={data_seq_len}, '
          f'model_seq_len={model_seq_len}, batch={batch_size}, steps={n_steps}, '
          f'pack_len={args.pack_len})')
sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                    drop_last=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
vocab_size = TokenizerVocabSize
model = NanoGPT(vocab_size, d_model, n_heads, n_layers, model_seq_len).to(device)
model = model.to(torch.bfloat16)

if not pathlib.Path(pretrained_path).exists():
    raise FileNotFoundError(f'pretrained checkpoint not found: {pretrained_path}')
chkp = torch.load(pretrained_path, map_location=device)
state_dict = {k.replace('module.', '', 1): v for k, v in chkp['model'].items()}
model.load_state_dict(state_dict)
if is_main:
    print(f'loaded pretrained model from {pretrained_path} (step {chkp.get("step", "?")})')

optimizer = optim.AdamW(model.parameters(), lr=lr_peak, weight_decay=weight_decay)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()) * (1 - 0.01) + 0.01

# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
last_step = 0
if pathlib.Path(ckpt_path).exists():
    if is_main:
        print('SFT checkpoint found, resuming...')
    chkp = torch.load(ckpt_path, map_location=device)
    sd = {k.replace('module.', '', 1): v for k, v in chkp['model'].items()}
    model.load_state_dict(sd)
    optimizer.load_state_dict(chkp['optimizer'])
    last_step = chkp['step']

for pg in optimizer.param_groups:
    pg.setdefault('initial_lr', lr_peak)
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda,
    last_epoch=-1 if last_step == 0 else last_step,
)

# DDP only pays off with more than one rank. With a single process it just
# adds a gradient-bucket copy — and, worse, it enqueues an NCCL ALLREDUCE every
# step, which turns any SIGSTOP pause into a watchdog timeout on resume
# ("Watchdog caught collective operation timeout ... ran for 29100395 ms").
# main.py and longcontext_ft.py already skip DDP this way.
if world_size > 1:
    model = DDP(model, device_ids=[local_rank])
raw_model = model.module if world_size > 1 else model

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
if is_main:
    print('SFT started')

# Stop cleanly on SIGTERM/SIGINT: finish the step, checkpoint, exit. Without
# this a `kill -TERM` loses every step since the last periodic save (and, before
# the DDP fix, could also trip the NCCL watchdog).
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
best_avg_loss = float('inf')
step = last_step
for epoch in range(1000):
    sampler.set_epoch(epoch)
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if args.pack_len > 0:
                # Packed windows have no padding, so no attention mask is
                # needed and the model's chunked-loss path can be used. That
                # avoids materialising logits for every position (at 4x8192
                # that is 3.3 GB of bf16 logits plus fp32 softmax temporaries),
                # which is what made the unpacked path run at 1.3k tok/s here
                # against 3.5k in the long-context trainer.
                loss, _, _ = model(inputs, targets=targets)
            else:
                attn_mask = (inputs != 50256)
                logits, _, _ = model(inputs, attn_mask=attn_mask)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    targets.reshape(-1),
                    ignore_index=-100,
                )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if len(loss_counting) >= 50:
            loss_counting = loss_counting[1:]
        loss_counting.append(loss.item())
        avg = sum(loss_counting) / len(loss_counting)

        if is_main:
            print(f"Step {step}: loss={loss.item():.4f} avgloss={avg:.4f}")
        if is_main and step % 200 == 0:
            torch.save({
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'avg_loss': avg,
            }, ckpt_path)

        dist.barrier()

        if stop_requested:
            break

        step += 1
        if step >= total_steps:
            break
    if step >= total_steps or stop_requested:
        break

if is_main:
    torch.save({
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
    }, ckpt_path)
    print('SFT model saved.')

dist.barrier()
dist.destroy_process_group()

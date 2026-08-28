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
from hyperparameters import *

# torchrun --nproc_per_node=N --master_addr=127.0.0.1 --master_port=29500 ./main.py

os.environ['AMD_SERIALIZE_KERNEL']='3'
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------
# torchrun sets: RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE,
#                MASTER_ADDR, MASTER_PORT
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
# Dataset
# ---------------------------------------------------------------------------
# Mixing weights (v2 corpus, ~38.5B tokens total):
#   slimpajama 14.1B - general web/book/wiki text (core knowledge)
#   cosmopedia 20.6B - synthetic educational text (reasoning-ish prose)
#   ultrachat    1.9B - chat format (if the .bin is present on this machine)
#   openthought  0.8B - reasoning traces w/ <think></think> tags (capability data)
dataset = MixedDatasetManager(device, [
    ('slimpajama', sorted(glob.glob('slimpajama_subset_*.bin')), 0.40),
    ('cosmopedia', sorted(glob.glob('cosmopediav2_*.bin')),      0.35),
    ('ultrachat',  ['UltraChat_uint16le.bin'],                    0.15),
    ('openthought', sorted(glob.glob('openthought_*.bin')),       0.10),
])
if is_main:
    print('training data loaded')
vocab_size = TokenizerVocabSize

# ---------------------------------------------------------------------------
# Model, optimizer, scheduler
# ---------------------------------------------------------------------------
model = NanoGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len).to(device)
model = model.to(torch.bfloat16)

optimizer = optim.AdamW(model.parameters(), lr=lr_peak, weight_decay=weight_decay)

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = max(0.0, min(1.0, progress))
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()) * (1 - 0.01) + 0.01

# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------
last_step = 0
if pathlib.Path('blacksmith-1_model.pt').exists():
    if is_main:
        print('previous checkpoint found. loading....')
    chkp = torch.load('blacksmith-1_model.pt', map_location=device)
    # Strip DDP "module." prefix if present (saved from a DDP-wrapped model).
    state_dict = {k.replace('module.', '', 1): v for k, v in chkp['model'].items()}
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(chkp['optimizer'])
    last_step = chkp['step']

# Scheduler must be created AFTER loading optimizer state so that
# last_epoch=last_step fast-forwards the internal LR to the right point.
# LambdaLR with last_epoch >= 0 requires 'initial_lr' in param_groups
# (it's normally set by the scheduler during a fresh init with last_epoch=-1).
for pg in optimizer.param_groups:
    pg.setdefault('initial_lr', lr_peak)
scheduler = optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda,
    last_epoch=-1 if last_step == 0 else last_step,
)

# Wrap in DDP AFTER loading checkpoint (DDP adds "module." prefix to keys,
# so we load the raw state dict first, then wrap).
model = DDP(model, device_ids=[local_rank])

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
if is_main:
    print('training started')

loss_counting = []
for step in range(last_step, min(last_step + n_steps, total_steps)):
    inputs, targets = dataset.get_batch(batch_size, max_seq_len)

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
        print(f"Step {step}: loss={loss.item():.4f} "
              f"avgloss={sum(loss_counting) / len(loss_counting):.4f}")
    if step % 50 == 0 and is_main:
        print('saving checkpoint...')
        # Save the unwrapped module's state_dict (no "module." prefix).
        torch.save({
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
        }, 'blacksmith-1_model.pt')
    # All ranks stay in sync at the save boundary.
    dist.barrier()

if is_main:
    torch.save({
        'model': model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
    }, 'blacksmith-1_model.pt')
    print('model saved.')

dist.barrier()
dist.destroy_process_group()

---
language:
- en
license: mit
pipeline_tag: text-generation
library_name: pytorch
tags:
- text-generation
- conversational
- long-context
- gguf
- from-scratch
- custom-architecture
- 522m
datasets:
- cerebras/SlimPajama-627B
- HuggingFaceTB/cosmopedia
- openbmb/UltraChat
- open-thoughts/OpenThoughts2-1M
- databricks/databricks-dolly-15k
- zai-org/LongAlign-10k
- hotpotqa/hotpot_qa
---

# Blacksmith-1a

Repository: `TriusAI/blacksmith-1a`

A 522M-parameter decoder-only transformer trained from scratch on a single
consumer iGPU (Radeon 8060S, 96 GiB VRAM). It is not a fine-tune of any
released model — the architecture, the data pipeline, the long-context
extension and the fine-tuning all live in `code/`.

**Honest status:** this is a small, coherent, well-formatted chat model with
almost no world knowledge. It gets its own identity right, answers "capital of
France" correctly about two thirds of the time, cannot do arithmetic, and
cannot retrieve information from more than roughly a thousand tokens back
despite a 64k trained window. Treat it as a research artifact, not a product.

## Architecture

| | |
|---|---|
| parameters | 522M (non-embedding ~430M) |
| layers | 20 |
| d_model / heads / head_dim | 1280 / 10 / 128 (MHA, no GQA) |
| FFN | 5120, GELU, no gate |
| norm | LayerNorm (eps 1e-5), pre-norm |
| positions | RoPE, theta 10000, **interleaved** pairing |
| context | trained 4,096 → 8k → 16k → 32k → **64k** |
| embeddings | untied input/output |
| tokenizer | GPT-2 BPE (50,257) + 4 control tokens → 50,261 |
| special tokens | `<\|im_start\|>`, `<\|im_end\|>`, `<think>`, `</think>`; EOT (50256) terminates a conversation |

The architecture is implemented in `code/model.py` as `NanoGPT`. It is **not** a
`transformers` architecture, so `AutoModel.from_pretrained` will not load it —
use the bundled code.

## Training

| stage | steps | tokens/step | final loss | notes |
|---|---|---|---|---|
| pretraining | 75,600 | 32 × 4096 | 1.97 | SlimPajama 40%, Cosmopedia v2 35%, UltraChat 15%, OpenThoughts 10%; ~9.9B tokens |
| long context 8k | 1,577 | 32 × 8192 | 1.99 | stopped when the average flattened |
| long context 16k | 1,014 | 16 × 16384 | 1.99 | |
| long context 32k | 956 | 8 × 32768 | 1.93 | |
| long context 64k | 277 | 4 × 65536 | ~2.1 | 421 s/step, `torch.compile` disabled |
| SFT | 3,000 | 4 × 8192 packed | 1.36 | UltraChat + LongAlign + HotpotQA + synthetic needle data |
| retrieval SFT | 2,500 | 4 × 8192 packed | 1.35 | 85% synthetic needle-in-a-haystack |
| identity polish | 2 × 300 | 4 × 8192 packed | 0.76 | identity at ~50% of tokens, `lr 5e-5` |

Long-context phases train at longer *positions* with `rope_scale = 1`; the
context is extended by training, not by scaling. YaRN is used only past the
trained length, at inference/export time.

## Usage

```python
import torch
from code.model import NanoGPT
from code.tokenmaker import Tokenizer, TokenizerVocabSize
from code.hyperparameters import d_model, n_heads, n_layers, max_seq_len

model = NanoGPT(TokenizerVocabSize, d_model, n_heads, n_layers, max_seq_len).cuda()
model = model.to(torch.bfloat16)
ckpt = torch.load('blacksmith-1a.pt', map_location='cuda', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
```

or run the bundled REPL:

```zsh
python3 code/generate.py --ckpt blacksmith-1a.pt
```

`code/tokenmaker.py` builds the tokenizer with
`GPT2Tokenizer.from_pretrained('gpt2')`; the equivalent files are also in this
repo (`tokenizer.json`), so no download is needed if you point that call at the
repo root. Set `HF_HUB_OFFLINE=1` once the gpt2 tokenizer is cached.

Prompt format (what the model was fine-tuned on):

```
<|endoftext|><|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
```

`chat_template.jinja` encodes the same thing for tools that consume HF chat
templates.

### GGUF (llama.cpp)

`gguf/blacksmith-1a-128k-f16.gguf` is an f16 export for llama.cpp's `gptneox`
architecture, with YaRN baked into the metadata. `gguf/blacksmith-1a-128k-Q4_K_M.gguf`
is the same file quantised to 327 MB (perplexity 23.21 vs 22.86 for f16 on the
same 512-token window, i.e. +1.5%):

```
gptneox.context_length                     = 131072
gptneox.rope.scaling.type                  = yarn
gptneox.rope.scaling.factor                = 2.0
gptneox.rope.scaling.original_context_length = 65536
```

```zsh
llama-cli -m gguf/blacksmith-1a-128k-Q4_K_M.gguf -c 131072   # 327 MB
llama-cli -m gguf/blacksmith-1a-128k-f16.gguf   -c 131072    # 1047 MB
```

The export was validated numerically: llama.cpp and the PyTorch model score the
same perplexity (19.33 vs 19.34) on a 512-token window, after three
conversion bugs were found and fixed (f16 biases, the missing `"gpt-2"`
pre-tokenizer hint, and the interleaved→half-split RoPE permutation). See
`docs/GGUF.md`.

## Evaluation

Sampled at temperature 0.7 with 2–4 samples per prompt; retrieval is greedy.

| prompt | result |
|---|---|
| "Who are you?" / "Hello! Who are you?" / "What's your name?" | **correct, all phrasings** ("I'm Blacksmith, a language model trained by Trius.") |
| "What is the capital of France?" | Paris — ~2–3 of 3 samples |
| "Why is the sky blue?" | names Rayleigh scattering — 1–3 of 3 |
| "Summarize the plot of Hamlet" | fluent and structured, factually invented |
| "Write a Python function that reverses a string" | plausible-looking code block, not valid Python |
| "What is 2+2?" | **never correct** |

Retrieval (needle-in-a-haystack, chat format, 3 samples per cell):

| context | adjacent | ~2k tokens back |
|---|---|---|
| 4,096 | 3/3 | 0–2 of 9 |
| 16,384 | 0–2 of 3 | 0/9 |
| 65,536 | 0/3 | 0/3 |

Training on 90M tokens of synthetic needle data (85% of the mix, exact
phrasing) moved 4k-context retrieval from 0/9 to 2/9 and left 16k at 0/9. The
model learns the answer *format* easily and the *lookup* only within roughly a
thousand tokens — a capacity limit at this parameter count, not a data one.

### Note on evaluation validity

All generation numbers above were taken **after** fixing a causal-mask bug in
`code/model.py`: the inference path previously prefilled prompts with unmasked
attention, so every position could see its own future. Training was never
affected, and GGUF inference was never affected (llama.cpp is causal by
construction). Numbers in `logs/` predate the fix.

## Limitations

- **No reliable world knowledge.** Paris is a coin flip; most factual questions
  produce confident inventions.
- **No arithmetic**, including single-digit sums.
- **Long context is capacity, not recall.** 64k training gives coherent
  continuation and no crash, but the model cannot look things up beyond ~1k
  tokens. Use RAG for retrieval.
- **Persona is brittle.** Identity survives the fine-tuning mix used here; a
  different mix can replace it with a confabulated persona within a few hundred
  steps.
- **English only**, GPT-2 tokenizer, no tool use, no safety tuning.

## Files

```
blacksmith-1a.pt                       final chat model, bf16, 3.0 GB
gguf/blacksmith-1a-128k-Q4_K_M.gguf    Q4_K_M GGUF, 327 MB, YaRN 2x from 64k
gguf/blacksmith-1a-128k-f16.gguf       f16 GGUF, 1047 MB (reference export)
code/                         model, training, evaluation and conversion scripts
config.json                   architecture description (custom, not transformers)
chat_template.jinja           chat template
tokenizer.json, tokenizer_config.json
docs/GGUF.md                  export details, metadata semantics, KV-cache budget
logs/                         training logs from the pretraining and SFT runs
```

Reproduce any stage with `code/` and the logs; the data pipeline is
`code/convert_datasets.py`, `code/make_niah.py`, `code/build_mix.py`, and the
pretokenisation script `code/prepare_ultra_chat.py`.

## License

Weights and code: **MIT**.

For transparency, the training corpora carry their own terms, and some are
share-alike or research-only:

- SlimPajama — ODC-By 1.0
- Cosmopedia v2 — ODC-By 1.0 (generated with Mixtral)
- UltraChat — MIT (generated with GPT-3.5/4)
- OpenThoughts — permissive, model-generated
- Databricks Dolly 15k — CC-BY-SA 3.0
- LongAlign-10k — research use (generated with Claude 2.1)
- HotpotQA — CC BY-SA 4.0

The MIT licence above covers the weights and the code in this repository, not
any upstream corpus; if you redistribute the data itself, check those terms.

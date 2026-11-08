#!/usr/bin/env python3
"""Export a Blacksmith NanoGPT checkpoint to GGUF for llama.cpp.

NanoGPT is not a Hugging Face architecture, so `convert_hf_to_gguf.py` cannot
read it. The closest llama.cpp architecture is `gptneox`: LayerNorm + GELU
(no gate) + NEOX-style RoPE + untied embeddings all match. The one mismatch is
that llama.cpp's gptneox loader *requires* bias tensors for every norm, the
fused qkv and the FFN, while NanoGPT has none. We therefore emit zero-filled
biases, which is numerically identical to having no bias.

Tensor mapping (torch shape -> gguf shape is reversed by GGUFWriter):

    W_embed [V,D]                -> token_embd.weight          [D,V]
    ln_f.weight/bias [D]         -> output_norm.weight/bias
    head.weight [V,D]            -> output.weight              [D,V]
    blk.N.ln1.weight/bias        -> blk.N.attn_norm.weight/bias
    blk.N.Wq/Wk/Wv [D,D]         -> blk.N.attn_qkv.weight      [D,3D] (concat)
    (none)                       -> blk.N.attn_qkv.bias        [3D]  zeros
    blk.N.Wo [D,D]               -> blk.N.attn_output.weight   [D,D]
    blk.N.ln2.weight/bias        -> blk.N.ffn_norm.weight/bias
    blk.N.fc1.weight [4D,D]      -> blk.N.ffn_up.weight        [D,4D]
    blk.N.fc2.weight [D,4D]      -> blk.N.ffn_down.weight      [4D,D]

Usage:

    python3 convert_to_gguf.py --ckpt blacksmith-1_model.pt --out blacksmith-1.gguf
    python3 convert_to_gguf.py --ckpt blacksmith-1_lc32k.pt \\
        --context-length 131072 --rope-scaling yarn --rope-factor 4 \\
        --rope-orig-ctx 32768 --out blacksmith-1-128k.gguf
"""
import argparse
import json
import math
import pathlib

import numpy as np
import torch

import gguf
from hyperparameters import d_model, n_heads, n_layers


# The SFT corpora (sft_dataset.py) format every turn as
#   <|im_start|>{role}\n{content}<|im_end|>\n
# and terminate a conversation with EOT. generate.py additionally puts an EOT
# *before* each user turn, which is just the previous conversation's
# terminator; a fresh chat therefore starts at <|im_start|>. The template
# below reproduces exactly that.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

EOS_ID = 50256  # <|endoftext|>; also the conversation terminator in the .bin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='blacksmith-1_model.pt',
                   help='checkpoint with a "model" state dict')
    p.add_argument('--out', default=None,
                   help='output .gguf path (default: <ckpt stem>.gguf)')
    p.add_argument('--n-heads', type=int, default=n_heads,
                   help='attention heads; not recoverable from the state dict')
    p.add_argument('--context-length', type=int, default=4096,
                   help='n_ctx_train to advertise. Use the length the model '
                        'was actually trained at unless YaRN extends it.')
    p.add_argument('--rope-scaling', choices=['none', 'yarn'], default='none')
    p.add_argument('--rope-factor', type=float, default=1.0,
                   help='context stretch factor for --rope-scaling yarn')
    p.add_argument('--rope-orig-ctx', type=int, default=None,
                   help='context the model was trained at before scaling '
                        '(default: --context-length / --rope-factor)')
    p.add_argument('--rope-attn-factor', type=float, default=None,
                   help='YaRN temperature, matching model.py: '
                        '0.1*ln(factor)+1 (default: computed from factor)')
    p.add_argument('--dtype', choices=['f16', 'bf16', 'f32'], default='f16')
    p.add_argument('--name', default=None)
    p.add_argument('--no-chat-template', action='store_true')
    return p.parse_args()


def load_state(path):
    chk = torch.load(path, map_location='cpu', weights_only=False)
    state = chk.get('model', chk)
    state = {k.replace('module.', '', 1): v for k, v in state.items()}
    return state, chk.get('step')


def to_np(t, dtype):
    return t.detach().to(torch.float32).numpy().astype(dtype)


def main():
    args = parse_args()
    ckpt = pathlib.Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f'checkpoint not found: {ckpt}')
    out = pathlib.Path(args.out or (ckpt.stem + '.gguf'))

    state, step = load_state(ckpt)
    np_dtype = {'f16': np.float16, 'bf16': np.float32, 'f32': np.float32}[args.dtype]

    # Derive the shape facts from the tensors themselves so a mismatched
    # hyperparameters.py cannot silently produce a broken file.
    vocab, d_embed = state['W_embed'].shape
    d_ff = state['blocks.0.ffn.fc1.weight'].shape[0]
    n_layer = 1 + max(int(k.split('.')[1]) for k in state if k.startswith('blocks.'))
    d_head = d_embed // args.n_heads
    if d_embed % args.n_heads:
        raise SystemExit(f'd_model {d_embed} not divisible by {args.n_heads} heads')
    print(f'{ckpt.name}: step={step} vocab={vocab} d_model={d_embed} '
          f'ffn={d_ff} layers={n_layer} heads={args.n_heads} head_dim={d_head}')

    if d_model != d_embed or n_layers != n_layer:
        print(f'  note: hyperparameters.py says d_model={d_model} '
              f'n_layers={n_layers}; using the checkpoint values')

    writer = gguf.GGUFWriter(str(out), 'gptneox')

    # ---- architecture -----------------------------------------------------
    writer.add_name(args.name or ckpt.stem)
    writer.add_context_length(args.context_length)
    writer.add_embedding_length(d_embed)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(d_ff)
    writer.add_head_count(args.n_heads)
    writer.add_head_count_kv(args.n_heads)  # MHA: no GQA in NanoGPT
    writer.add_rope_dimension_count(d_head)
    writer.add_rope_freq_base(10000.0)
    # NanoGPT uses the sequential residual (x + attn; x + ffn), not GPT-NeoX's
    # parallel one. model.py: x = x + attn; x = x + ffn.
    writer.add_parallel_residual(False)
    # nn.LayerNorm default; model.py never overrides eps.
    writer.add_layer_norm_eps(1e-5)
    writer.add_file_type(1 if args.dtype == 'f16' else 0)  # 1 = MOSTLY_F16

    if args.rope_scaling == 'yarn':
        factor = args.rope_factor
        if factor <= 1.0:
            raise SystemExit('--rope-factor must be > 1 for yarn scaling')
        orig = args.rope_orig_ctx or int(round(args.context_length / factor))
        attn_factor = (args.rope_attn_factor
                       if args.rope_attn_factor is not None
                       else 0.1 * math.log(factor) + 1.0)
        writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
        writer.add_rope_scaling_factor(float(factor))
        writer.add_rope_scaling_orig_ctx_len(int(orig))
        # finetuned=false: the weights never saw YaRN during training, so the
        # extension is extrapolation. llama.cpp only uses this for reporting.
        writer.add_rope_scaling_finetuned(False)
        writer.add_rope_scaling_attn_factors(float(attn_factor))
        print(f'  yarn: factor={factor} orig_ctx={orig} attn_factor={attn_factor:.4f} '
              f'-> effective {orig * factor:.0f}')
    else:
        writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

    # ---- tokenizer --------------------------------------------------------
    from tokenmaker import Tokenizer, TokenizerVocabSize
    if TokenizerVocabSize != vocab:
        raise SystemExit(f'tokenizer vocab {TokenizerVocabSize} != checkpoint '
                         f'vocab {vocab}')

    tokens = [Tokenizer.convert_ids_to_tokens(i) for i in range(vocab)]
    types = [gguf.TokenType.NORMAL] * vocab
    for i in range(50257, vocab):
        types[i] = gguf.TokenType.CONTROL

    merges_path = pathlib.Path(getattr(Tokenizer, 'merges_file', '') or '.')
    if not merges_path.is_file():
        # Some transformers versions drop GPT2Tokenizer.merges_file; the file
        # still sits next to vocab.json in the HF snapshot.
        merges_path = pathlib.Path(Tokenizer.vocab_file).parent / 'merges.txt'
    if not merges_path.is_file():
        raise SystemExit(f'GPT-2 merges file not found (looked at {merges_path})')
    merges = [ln.rstrip('\n') for ln in merges_path.open(encoding='utf-8')
              if ln.strip() and not ln.startswith('#version')]

    writer.add_tokenizer_model('gpt2')
    # Without the pretokenizer hint llama.cpp falls back to LLAMA_VOCAB_PRE_TYPE_DEFAULT,
    # which splits digit/space runs differently from GPT-2's regex: " 31" came out
    # as ['Ġ','31'] instead of ['Ġ31'], inflating a 512-token text to 535 tokens.
    writer.add_tokenizer_pre('gpt-2')  # llama.cpp's name for LLAMA_VOCAB_PRE_TYPE_GPT2
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(merges)
    writer.add_eos_token_id(EOS_ID)
    writer.add_bos_token_id(EOS_ID)
    writer.add_pad_token_id(EOS_ID)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    if not args.no_chat_template:
        writer.add_chat_template(CHAT_TEMPLATE)
    print(f'  tokenizer: {len(tokens)} tokens ({vocab - 50257} added), '
          f'{len(merges)} merges')

    # ---- tensors ----------------------------------------------------------
    def add(name, tensor):
        # 1-D tensors (norms and biases) must be f32: llama.cpp's CPU backend
        # builds f32 activations and rejects an f16 bias in ggml_add
        # ("binary_op: unsupported types: dst: f32, src0: f32, src1: f16").
        # This is what the upstream converters do too.
        arr = to_np(tensor, np.float32 if tensor.ndim == 1 else np_dtype)
        writer.add_tensor(name, arr)

    def rope_permute(w):
        """Reorder a q/k projection from interleaved to half-split rope dims.

        model.py's apply_rope rotates the pairs (x[2i], x[2i+1]) — the
        interleaved / "GPT-NeoX original" convention. llama.cpp's
        LLAMA_ROPE_TYPE_NEOX (used for the gptneox arch) instead rotates the
        pairs (x[i], x[i + d_head/2]). The two are the same rotation viewed in
        different orders, so permuting the output dims of Wq/Wk per head
        converts one into the other; Q·K and therefore the attention output
        are unchanged. Verified numerically: without this permutation
        llama.cpp reproduces a model that scores PPL 135 instead of 19 on the
        same tokens, with this permutation the two agree.
        """
        d = w.shape[0] // args.n_heads
        idx = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
        if w.ndim == 1:
            return w.reshape(args.n_heads, d)[:, idx].reshape(-1)
        return w.reshape(args.n_heads, d, *w.shape[1:])[:, idx].reshape(w.shape)

    add('token_embd.weight', state['W_embed'])
    add('output_norm.weight', state['ln_f.weight'])
    add('output_norm.bias', state['ln_f.bias'])
    add('output.weight', state['head.weight'])
    # NOTE: model.py's `head` is an nn.Linear with a bias, and llama.cpp's
    # gptneox graph applies `output` without one. The dropped bias shifts each
    # token's logit by at most 0.067 (measured on the step-75600 checkpoint),
    # which leaves top-1 and top-5 unchanged and gives KL(p||q) ~ 4e-5, so it
    # is safe to omit. It is reported here rather than silently ignored.
    if 'head.bias' in state:
        bias_max = float(state['head.bias'].abs().max())
        print(f'  dropping head.bias (max |value| = {bias_max:.4f}; '
              f'gptneox has no output bias)')

    for i in range(n_layer):
        b = f'blocks.{i}.'
        add(f'blk.{i}.attn_norm.weight', state[b + 'ln1.weight'])
        add(f'blk.{i}.attn_norm.bias', state[b + 'ln1.bias'])

        # llama.cpp reshapes the fused tensor to
        # (head_dim, n_head + 2*n_head_kv, n_tokens), so the row blocks must be
        # [all Q heads][all K heads][all V heads].
        # Q and K also need the interleaved -> half-split rope permutation;
        # V is untouched because attention output is a sum over V.
        wqkv = torch.cat([rope_permute(state[b + 'attn.Wq.weight']),
                          rope_permute(state[b + 'attn.Wk.weight']),
                          state[b + 'attn.Wv.weight']], dim=0)
        bqkv = torch.cat([rope_permute(state[b + 'attn.Wq.bias']),
                          rope_permute(state[b + 'attn.Wk.bias']),
                          state[b + 'attn.Wv.bias']], dim=0)
        add(f'blk.{i}.attn_qkv.weight', wqkv)
        add(f'blk.{i}.attn_qkv.bias', bqkv)

        add(f'blk.{i}.attn_output.weight', state[b + 'attn.Wo.weight'])
        add(f'blk.{i}.attn_output.bias', state[b + 'attn.Wo.bias'])

        add(f'blk.{i}.ffn_norm.weight', state[b + 'ln2.weight'])
        add(f'blk.{i}.ffn_norm.bias', state[b + 'ln2.bias'])

        add(f'blk.{i}.ffn_up.weight', state[b + 'ffn.fc1.weight'])
        add(f'blk.{i}.ffn_up.bias', state[b + 'ffn.fc1.bias'])
        add(f'blk.{i}.ffn_down.weight', state[b + 'ffn.fc2.weight'])
        add(f'blk.{i}.ffn_down.bias', state[b + 'ffn.fc2.bias'])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size = out.stat().st_size
    print(f'wrote {out} ({size / 1e6:.0f} MB, {args.dtype})')


if __name__ == '__main__':
    main()

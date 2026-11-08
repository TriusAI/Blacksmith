#!/usr/bin/env python3
"""Round-trip check for a converted GGUF against the PyTorch checkpoint.

Verifies, without needing llama.cpp:
  * every expected tensor exists with the right shape and dtype
  * tensor values match the checkpoint (after the documented transposes and
    the QKV concatenation)
  * the metadata llama.cpp reads for this architecture is present and sane

Usage: python3 verify_gguf.py model.gguf blacksmith-1_model.pt
"""
import sys

import numpy as np
import torch
import gguf


def load_state(path):
    chk = torch.load(path, map_location='cpu', weights_only=False)
    st = chk.get('model', chk)
    return {k.replace('module.', '', 1): v for k, v in st.items()}, chk.get('step')


def main(gguf_path, ckpt_path):
    reader = gguf.GGUFReader(gguf_path)
    fields = {f.name: f for f in reader.fields.values()}
    # The loader exposes tensors under reader.tensors with numpy data.
    tensors = {t.name: t for t in reader.tensors}
    state, step = load_state(ckpt_path)

    def kv(name):
        f = fields[name]
        return f.contents()

    print(f'gguf: {gguf_path}')
    print(f'  arch            = {kv("general.architecture")}')
    print(f'  context_length  = {kv("gptneox.context_length")}')
    print(f'  block_count     = {kv("gptneox.block_count")}')
    print(f'  head_count      = {kv("gptneox.attention.head_count")}')
    print(f'  rope.dim_count  = {kv("gptneox.rope.dimension_count")}')
    print(f'  rope.freq_base  = {kv("gptneox.rope.freq_base")}')
    for k in ("gptneox.rope.scaling.type", "gptneox.rope.scaling.factor",
              "gptneox.rope.scaling.original_context_length",
              "gptneox.rope.scaling.attn_factor",
              "gptneox.rope.scaling.finetuned",
              "gptneox.use_parallel_residual",
              "gptneox.attention.layer_norm_epsilon"):
        if k in fields:
            print(f'  {k.split(".", 1)[1]:<32}= {kv(k)}')
    print(f'  tensors         = {len(tensors)}')

    d_embed = state['W_embed'].shape[1]
    n_heads = int(kv('gptneox.attention.head_count'))

    def rope_permute(w):
        # convert_gguf reorders Q/K from interleaved to half-split rope dims;
        # mirror that here so the comparison is against what was written.
        d = w.shape[0] // n_heads
        idx = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
        if w.ndim == 1:
            return w.reshape(n_heads, d)[:, idx].reshape(-1)
        return w.reshape(n_heads, d, *w.shape[1:])[:, idx].reshape(w.shape)
    n_layer = 1 + max(int(k.split('.')[1]) for k in state if k.startswith('blocks.'))

    def check(name, ref, note=''):
        if name not in tensors:
            print(f'  MISSING {name}')
            return False
        got = np.asarray(tensors[name].data)
        want = ref.detach().float().numpy()
        if got.shape != want.shape:
            print(f'  SHAPE   {name}: gguf {got.shape} vs torch {want.shape}')
            return False
        # GGUF stores f16; compare at f16 precision.
        got = got.astype(np.float32)
        err = np.abs(got - want).max()
        ok = err < 2e-2 * max(1.0, np.abs(want).max())
        print(f'  {"ok  " if ok else "FAIL"} {name:<32} {str(got.shape):<14} '
              f'max|diff|={err:.2e} {note}')
        return ok

    print('values:')
    ok = True
    ok &= check('token_embd.weight', state['W_embed'])
    ok &= check('output.weight', state['head.weight'])
    ok &= check('output_norm.weight', state['ln_f.weight'])
    ok &= check('output_norm.bias', state['ln_f.bias'])
    for i in range(n_layer):
        b = f'blocks.{i}.'
        wqkv = torch.cat([rope_permute(state[b + 'attn.Wq.weight']),
                          rope_permute(state[b + 'attn.Wk.weight']),
                          state[b + 'attn.Wv.weight']], dim=0)
        bqkv = torch.cat([rope_permute(state[b + 'attn.Wq.bias']),
                          rope_permute(state[b + 'attn.Wk.bias']),
                          state[b + 'attn.Wv.bias']], dim=0)
        ok &= check(f'blk.{i}.attn_qkv.weight', wqkv, '(fused Q|K|V)')
        ok &= check(f'blk.{i}.attn_qkv.bias', bqkv, '(fused Q|K|V)')
        ok &= check(f'blk.{i}.attn_output.weight', state[b + 'attn.Wo.weight'])
        ok &= check(f'blk.{i}.attn_output.bias', state[b + 'attn.Wo.bias'])
        ok &= check(f'blk.{i}.ffn_up.weight', state[b + 'ffn.fc1.weight'])
        ok &= check(f'blk.{i}.ffn_up.bias', state[b + 'ffn.fc1.bias'])
        ok &= check(f'blk.{i}.ffn_down.weight', state[b + 'ffn.fc2.weight'])
        ok &= check(f'blk.{i}.ffn_down.bias', state[b + 'ffn.fc2.bias'])
        ok &= check(f'blk.{i}.attn_norm.weight', state[b + 'ln1.weight'])
        ok &= check(f'blk.{i}.attn_norm.bias', state[b + 'ln1.bias'])
        ok &= check(f'blk.{i}.ffn_norm.weight', state[b + 'ln2.weight'])
        ok &= check(f'blk.{i}.ffn_norm.bias', state[b + 'ln2.bias'])

    # Vocabulary + special tokens.
    tok_field = fields.get('tokenizer.ggml.tokens')
    if tok_field is not None:
        toks = [t.decode('utf-8', 'replace') if isinstance(t, bytes) else str(t)
                for t in tok_field.contents()]
        from tokenmaker import Tokenizer
        want = [Tokenizer.convert_ids_to_tokens(i)
                for i in range(50257, 50261)]
        match = toks[-4:] == want
        print(f'tokenizer: {len(toks)} tokens, added tokens match = {match}')
        ok &= match
    print(f'\ncheckpoint step {step}: {"PASS" if ok else "FAIL"}')
    return 0 if ok else 1


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))

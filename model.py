import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def compute_rope_tables(max_seq_len, d_head, device, rope_scale=1.0,
                        original_max_pos=4096, beta_fast=32, beta_slow=1):
    """
    Compute YaRN-modulated RoPE cos/sin tables.

    YaRN (Yet another RoPE extensioN, Peng et al. 2023) extends RoPE beyond
    its training length by interpolating only the LOW-frequency dimensions
    (which encode long-range position info) while leaving HIGH-frequency
    dimensions (short-range/local info) untouched. A temperature factor
    sqrt(1/t) partially compensates for the attention-entropy change that
    interpolation causes.

    rope_scale: position stretch factor. Positions beyond original_max_pos
        are mapped to original_max_pos * rope_scale effectively; the
        low-frequency dims get interpolated by 1/rope_scale.
    original_max_pos: the context length the model was TRAINED at. Positions
        below this are left unscaled (no quality loss on short contexts).
    """
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, dtype=torch.float) / d_head))

    if rope_scale != 1.0:
        # YaRN ramp (Peng et al. 2023, section 3.4.1): the ramp is driven by
        # the number of rotations a dimension completes over the ORIGINAL
        # training context: h = original_max_pos * inv_freq / (2*pi).
        # - h >= beta_fast (rotates many times): high-frequency, encodes
        #   local/short-range info -> keep frequency untouched (ramp = 0).
        # - h <= beta_slow (barely rotates): low-frequency, encodes long-range
        #   position -> interpolate frequency by 1/rope_scale (ramp = 1).
        def ramp(h):
            if h >= beta_fast:
                return 0.0
            if h <= beta_slow:
                return 1.0
            return (beta_fast - h) / (beta_fast - beta_slow)

        rotations = original_max_pos * inv_freq / (2 * torch.pi)
        low_frac = torch.tensor(
            [ramp(h.item()) for h in rotations],
            dtype=torch.float,
        )
        # Interpolate frequencies: fast dims unchanged, slow dims divided by scale.
        inv_freq = inv_freq * (1 - low_frac) + (inv_freq / rope_scale) * low_frac

    pos = torch.arange(max_seq_len, dtype=torch.float)
    angles = pos[:, None] * inv_freq[None, :]
    # Temperature factor compensates for entropy change from interpolation.
    if rope_scale != 1.0:
        temp = 0.1 * math.log(rope_scale) + 1.0
    else:
        temp = 1.0
    cos = torch.cos(angles) / math_sqrt(temp)
    sin = torch.sin(angles) / math_sqrt(temp)
    return cos, sin


def math_sqrt(x):
    return x ** 0.5


def rope(x, pos):
    # x :: (batch, n_heads, seq, d_head)
    # pos :: (seq,) absolute position indices
    d_head = x.shape[-1]
    freqs = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, device=x.device, dtype=x.dtype) / d_head))
    angles = pos[:, None] * freqs[None, :]  # (seq, d_head/2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, rope_scale=1.0,
                 original_max_pos=4096):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, n_heads * self.d_head)
        self.Wk = nn.Linear(d_model, n_heads * self.d_head)
        self.Wv = nn.Linear(d_model, n_heads * self.d_head)
        self.Wo = nn.Linear(n_heads * self.d_head, d_model)

        # Precompute RoPE cos/sin tables for positions [0, max_seq_len),
        # optionally YaRN-scaled for context extension.
        d_head = self.d_head
        cos, sin = compute_rope_tables(
            max_seq_len, d_head, 'cpu',
            rope_scale=rope_scale, original_max_pos=original_max_pos,
        )
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def apply_rope(self, x, pos_offset, n_token):
        # x :: (batch, n_heads, seq, d_head)
        if pos_offset + n_token <= self.rope_cos.shape[0]:
            cos = self.rope_cos[pos_offset:pos_offset + n_token]
            sin = self.rope_sin[pos_offset:pos_offset + n_token]
        else:
            # Fall back to dynamic computation if positions exceed precomputed range.
            d_head = x.shape[-1]
            pos = torch.arange(n_token, device=x.device) + pos_offset
            freqs = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, device=x.device, dtype=x.dtype) / d_head))
            angles = pos[:, None] * freqs[None, :]
            cos = torch.cos(angles)
            sin = torch.sin(angles)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

    def forward(self, x, k_cache=None, v_cache=None, infer=False, pos_offset=0, attn_mask=None):
        batch, n_token, d_model = x.shape
        n_heads, d_head = self.n_heads, self.d_head

        Q = self.Wq(x).reshape(batch, n_token, n_heads, d_head).transpose(1, 2)
        K = self.Wk(x).reshape(batch, n_token, n_heads, d_head).transpose(1, 2)
        V = self.Wv(x).reshape(batch, n_token, n_heads, d_head).transpose(1, 2)

        Q = self.apply_rope(Q, pos_offset, n_token)
        K = self.apply_rope(K, pos_offset, n_token)

        if k_cache is not None:
            K = torch.cat([k_cache, K], dim=2)
            V = torch.cat([v_cache, V], dim=2)

        # attn_mask: (batch, seq) bool, True = real token, False = padding.
        # PyTorch's scaled_dot_product_attention attn_mask convention:
        #   boolean: True = ALLOWED (attended), False = masked (excluded).
        # We use the boolean form instead of the additive float form to
        # avoid bfloat16 overflow (finfo.min + finfo.min = -inf -> NaN).
        if infer:
            if attn_mask is not None:
                pad = attn_mask[:, None, None, :].to(torch.bool)
                out = F.scaled_dot_product_attention(Q, K, V, attn_mask=pad)
            else:
                out = F.scaled_dot_product_attention(Q, K, V)
        else:
            if attn_mask is None:
                out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
            else:
                T = Q.shape[-2]
                Tk = K.shape[-2]
                causal = torch.ones(T, Tk, dtype=torch.bool, device=Q.device).tril()
                pad = attn_mask[:, None, None, :].to(torch.bool)
                # A key position is attended iff causal allows it (on/below
                # diagonal) AND it's a real token (not padding).
                combined = causal[None, None, :, :] & pad
                out = F.scaled_dot_product_attention(Q, K, V, attn_mask=combined)
        out = out.transpose(1, 2).reshape(batch, n_token, n_heads * d_head)
        return self.Wo(out), K, V


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, rope_scale=1.0,
                 original_max_pos=4096):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_seq_len,
                                       rope_scale=rope_scale,
                                       original_max_pos=original_max_pos)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)

    def forward(self, x, k_cache=None, v_cache=None, infer=False, pos_offset=0, attn_mask=None):
        attn, k_new, v_new = self.attn(self.ln1(x), k_cache=k_cache, v_cache=v_cache,
                                       infer=infer, pos_offset=pos_offset, attn_mask=attn_mask)
        x = x + attn
        x = x + self.ffn(self.ln2(x))
        return x, k_new, v_new


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len,
                 rope_scale=1.0, original_max_pos=4096):
        super().__init__()
        self.W_embed = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len,
                             rope_scale=rope_scale,
                             original_max_pos=original_max_pos)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        # Positions per loss chunk. 1024 positions x 50261 vocab = ~0.4GB
        # bf16 logits per chunk — small enough to keep several alive during
        # autograd without threatening VRAM.
        self.loss_chunk_positions = 1024

    def forward(self, token_ids, k_cache=None, v_cache=None, infer=False, pos_offset=0, attn_mask=None, targets=None):
        x = self.W_embed[token_ids]
        if not infer:
            # Training path: gradient checkpointing + skip building KV caches
            # to save activation memory.
            for block in self.blocks:
                x, _, _ = checkpoint(block, x, attn_mask=attn_mask, use_reentrant=False)
            x = self.ln_f(x)
            if targets is not None:
                # Chunked + checkpointed loss. The full logits tensor for a
                # 50k vocab at long context dwarfs the model itself
                # (e.g. batch 48 x 4096 x 50261 = 19.8GB bf16, and
                # cross_entropy's internal fp32 log-softmax doubles that).
                # Merely splitting into chunks does NOT help: autograd
                # retains every chunk's logits + saved softmax until
                # backward, so total memory is unchanged. Wrapping each
                # chunk in its own checkpoint discards the chunk's logits
                # after accumulating its loss and recomputes them during
                # backward — only ONE chunk's logits are alive at any time.
                def chunk_loss(x_chunk, targets_chunk):
                    logits_chunk = self.head(x_chunk)
                    return F.cross_entropy(
                        logits_chunk.reshape(-1, logits_chunk.shape[-1]),
                        targets_chunk.reshape(-1),
                        reduction='sum',
                        ignore_index=-100,
                    )

                total_loss = 0.0
                chunk = self.loss_chunk_positions
                n = token_ids.shape[1]
                for start in range(0, n, chunk):
                    end = min(start + chunk, n)
                    total_loss = total_loss + checkpoint(
                        chunk_loss, x[:, start:end], targets[:, start:end],
                        use_reentrant=False,
                    )
                # Match F.cross_entropy's default reduction='mean' with
                # ignore_index: divide by the count of non-ignored targets.
                n_valid = (targets != -100).sum()
                return total_loss / n_valid, None, None
            logits = self.head(x)
            return logits, None, None

        k_caches = []
        v_caches = []
        for i, block in enumerate(self.blocks):
            x, k, v = block(
                x,
                k_cache=k_cache[i] if k_cache is not None else None,
                v_cache=v_cache[i] if v_cache is not None else None,
                infer=infer,
                pos_offset=pos_offset
            )
            k_caches.append(k)
            v_caches.append(v)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, k_caches, v_caches
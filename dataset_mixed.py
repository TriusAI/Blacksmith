import os

import torch
import numpy as np


class MixedDatasetManager:
    """
    Dataset manager that mixes multiple sources of pre-tokenized uint16
    little-endian .bin files.

    Each "source" is a group of .bin files treated as one logical corpus.
    A mixing weight controls how often each source is sampled per batch.

    Example:
        ds = MixedDatasetManager(device, [
            ('cosmopedia', ['cosmopediav2_00000000.bin', ...], 0.85),
            ('ultrachat',  ['UltraChat_uint16le.bin'],                 0.15),
        ])
        x, y = ds.get_batch(32, 1024)

    All files within a source are concatenated logically (offsets are
    remapped across file boundaries), so a window never spans two files
    of the same source — but it CAN span a document boundary inside a
    file (the EOT tokens sprinkled in by prepare_*.py act as soft
    separators; the model learns to recover across them, same as
    dataset2.py).
    """

    def __init__(self, device, sources):
        """
        sources: list of (name, files, weight) tuples.
            name   : str, label for logging
            files  : list of str, paths to .bin files
            weight  : float, relative sampling probability (need not sum to 1)
        """
        self.device = device
        self.sources = []

        total_weight = sum(w for _, _, w in sources)
        if total_weight <= 0:
            raise ValueError('total mixing weight must be positive')

        for name, files, weight in sources:
            if not files:
                raise ValueError(f'source "{name}" has no files')
            memmaps = []
            lengths = []
            cum = [0]
            skipped = []
            for f in files:
                if not os.path.exists(f):
                    skipped.append(f)
                    continue
                m = np.memmap(f, dtype=np.uint16, mode='r')
                memmaps.append(m)
                lengths.append(len(m))
                cum.append(cum[-1] + len(m))
            if skipped and len(memmaps) == 0:
                raise FileNotFoundError(
                    f'source "{name}": none of its files exist '
                    f'(missing: {skipped})'
                )
            self.sources.append({
                'name': name,
                'memmaps': memmaps,
                'lengths': np.array(lengths, dtype=np.int64),
                'cum': np.array(cum, dtype=np.int64),
                'total': cum[-1],
                'weight': weight / total_weight,
                'missing': skipped,
            })

        self.weights = torch.tensor([s['weight'] for s in self.sources],
                                    dtype=torch.float, device=device)

    def _sample_window(self, source, seq_len):
        """Sample one (x, y) window of length seq_len from a source."""
        total = source['total']
        if total < seq_len + 2:
            raise ValueError(
                f'source "{source["name"]}" ({total} tokens) too small '
                f'for seq_len={seq_len}'
            )
        # Global offset within the source's logical concatenation.
        s = int(torch.randint(0, total - seq_len - 1, (1,)).item())
        # Find which file this offset falls into.
        fidx = int(np.searchsorted(source['cum'], s, side='right')) - 1
        local = s - source['cum'][fidx]
        m = source['memmaps'][fidx]
        # `s` is bounded by the *source* total, so a draw can land within
        # seq_len of the next file's first token. Slicing there yields a
        # short window, and the mixed batch then dies in np.stack with
        # "all input arrays must have the same shape". Re-draw inside this
        # file so every window stays within a single file.
        if local + seq_len + 1 > len(m):
            local = int(torch.randint(0, len(m) - seq_len - 1, (1,)).item())
        x = np.array(m[local:local + seq_len], dtype=np.int64)
        y = np.array(m[local + 1:local + seq_len + 1], dtype=np.int64)
        return x, y

    def get_batch(self, batch_size, seq_len):
        # Pick which source each slot in the batch draws from.
        src_idx = torch.multinomial(self.weights, batch_size, replacement=True)
        xs, ys = [], []
        for i in src_idx.tolist():
            x, y = self._sample_window(self.sources[i], seq_len)
            xs.append(x)
            ys.append(y)
        x = torch.tensor(np.stack(xs), dtype=torch.long, device=self.device)
        y = torch.tensor(np.stack(ys), dtype=torch.long, device=self.device)
        return x, y
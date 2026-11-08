import json
import torch
import logging
from array import array
from torch.utils.data import Dataset
from tokenmaker import Tokenizer

logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)

IM_START = Tokenizer.encode('<|im_start|>')[0]
IM_END = Tokenizer.encode('<|im_end|>')[0]
EOT = Tokenizer.eos_token_id  # 50256
NEWLINE = Tokenizer.encode('\n')[0]
PAD_ID = EOT


def encode_turn(role, content):
    prefix = Tokenizer.encode(f'<|im_start|>{role}\n')
    body = Tokenizer.encode(content)
    suffix = [IM_END, NEWLINE]
    return prefix, body, suffix


def encode_conversation(turns):
    token_ids = []
    labels = []
    for role, content in turns:
        prefix, body, suffix = encode_turn(role, content)
        token_ids.extend(prefix)
        token_ids.extend(body)
        token_ids.extend(suffix)
        if role == 'assistant':
            labels.extend([-100] * len(prefix))
            labels.extend(body)
            labels.extend(suffix)
        else:
            labels.extend([-100] * (len(prefix) + len(body) + len(suffix)))
    token_ids.append(EOT)
    labels.append(EOT)
    return token_ids, labels


class SFTDataset(Dataset):
    def __init__(self, sources, max_seq_len=1024, drop_truncated=True):
        self.max_seq_len = max_seq_len
        self.samples = []
        for source in sources:
            path = source[0]
            repeat = source[1] if len(source) > 1 else 1
            file_samples = []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    messages = obj.get('data') or obj.get('messages')
                    if not messages:
                        continue
                    turns = []
                    for i, m in enumerate(messages):
                        role = 'user' if i % 2 == 0 else 'assistant'
                        turns.append((role, m))

                    encoded_turns = []
                    total_len = 0
                    for role, content in turns:
                        prefix, body, suffix = encode_turn(role, content)
                        tlen = len(prefix) + len(body) + len(suffix)
                        encoded_turns.append((role, prefix, body, suffix, tlen))
                        total_len += tlen

                    if total_len <= max_seq_len:
                        token_ids = []
                        labels = []
                        for role, prefix, body, suffix, _ in encoded_turns:
                            token_ids.extend(prefix)
                            token_ids.extend(body)
                            token_ids.extend(suffix)
                            if role == 'assistant':
                                labels.extend([-100] * len(prefix))
                                labels.extend(body)
                                labels.extend(suffix)
                            else:
                                labels.extend([-100] * (len(prefix) + len(body) + len(suffix)))
                    elif drop_truncated:
                        # Keep the last turns that fit; drop from the front
                        # so the final assistant response stays intact.
                        kept = []
                        kept_len = 0
                        for role, prefix, body, suffix, tlen in reversed(encoded_turns):
                            if kept_len + tlen > max_seq_len:
                                break
                            kept.append((role, prefix, body, suffix))
                            kept_len += tlen
                        kept.reverse()
                        # Must start with a user turn for a valid conversation.
                        if not kept or kept[0][0] != 'user':
                            continue
                        token_ids = []
                        labels = []
                        for role, prefix, body, suffix in kept:
                            token_ids.extend(prefix)
                            token_ids.extend(body)
                            token_ids.extend(suffix)
                            if role == 'assistant':
                                labels.extend([-100] * len(prefix))
                                labels.extend(body)
                                labels.extend(suffix)
                            else:
                                labels.extend([-100] * (len(prefix) + len(body) + len(suffix)))
                    else:
                        continue

                    # Terminate the conversation with EOT, matching the
                    # pretraining corpus (prepare_ultra_chat.py writes one EOT
                    # after every conversation). The terminator itself carries
                    # no assistant label, but having it in the sequence is what
                    # teaches the model to emit EOT at the end of a reply.
                    token_ids.append(EOT)
                    # The label shift below makes labels[i] the target for
                    # position i-1, so the terminator's own label is what
                    # teaches the model to emit EOT after the last turn.
                    labels.append(EOT)
                    # A conversation that already fills max_seq_len would grow
                    # to max_seq_len+1 here and break the collate stack; drop
                    # the terminator in that case.
                    if len(token_ids) > self.max_seq_len:
                        token_ids = token_ids[:self.max_seq_len]
                        labels = labels[:self.max_seq_len]

                    shifted = []
                    for i in range(len(token_ids) - 1):
                        shifted.append(labels[i + 1])
                    shifted.append(-100)
                    labels = shifted

                    file_samples.append((token_ids, labels))

            for _ in range(repeat):
                self.samples.extend(file_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token_ids, labels = self.samples[idx]
        pad_len = self.max_seq_len - len(token_ids)
        input_ids = token_ids + [PAD_ID] * pad_len
        labels = labels + [-100] * pad_len
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def encode_messages(messages, max_len):
    """One conversation -> (token_ids, labels), truncated from the front.

    Shared by PackedSFTDataset; mirrors the inline logic in SFTDataset so the
    two cannot drift apart in masking.
    """
    turns = []
    for i, content in enumerate(messages):
        role = 'user' if i % 2 == 0 else 'assistant'
        prefix, body, suffix = encode_turn(role, content)
        turns.append((role, prefix, body, suffix,
                      len(prefix) + len(body) + len(suffix)))
    total = sum(t[4] for t in turns)
    if total > max_len:
        kept, kept_len = [], 0
        for t in reversed(turns):
            if kept_len + t[4] > max_len:
                break
            kept.append(t)
            kept_len += t[4]
        kept.reverse()
        if not kept or kept[0][0] != 'user':
            return None
        turns = kept

    ids, labels = [], []
    for role, prefix, body, suffix, _ in turns:
        ids.extend(prefix)
        ids.extend(body)
        ids.extend(suffix)
        if role == 'assistant':
            labels.extend([-100] * len(prefix))
            labels.extend(body)
            labels.extend(suffix)
        else:
            labels.extend([-100] * (len(prefix) + len(body) + len(suffix)))
    ids.append(EOT)
    labels.append(EOT)
    shifted = labels[1:] + [-100]
    return ids, shifted


class PackedSFTDataset(Dataset):
    """Conversations concatenated into fixed-length windows (LongAlign packing).

    Fine-tuning on <=2048-token samples is what cost the step-6000 SFT its
    64k retrieval ability. Packing keeps the model seeing long positions while
    still training on chat data: each window is `pack_len` tokens of
    back-to-back conversations, with the loss masked to assistant tokens only.

    The stream is stored as array('i') rather than Python lists: at ~400M
    tokens the list version costs ~14 GB, the array version ~3 GB.
    """

    def __init__(self, sources, pack_len=8192, max_tokens=None, verbose=True):
        self.pack_len = pack_len
        ids, labels = array('i'), array('i')
        kept = dropped = 0
        for path, repeat in sources:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    messages = obj.get('data') or obj.get('messages')
                    if not messages:
                        continue
                    enc = encode_messages(messages, pack_len)
                    if enc is None:
                        dropped += 1
                        continue
                    for _ in range(repeat):
                        ids.extend(enc[0])
                        labels.extend(enc[1])
                        kept += 1
                    if max_tokens and len(ids) >= max_tokens:
                        break
            if max_tokens and len(ids) >= max_tokens:
                break

        if max_tokens:
            del ids[max_tokens:]
            del labels[max_tokens:]
        self.n = len(ids) // pack_len
        del ids[(self.n * pack_len):]
        del labels[(self.n * pack_len):]
        self.ids, self.labels = ids, labels
        if verbose:
            print(f'packed: {kept} conversations, {dropped} dropped, '
                  f'{len(ids)} tokens -> {self.n} windows of {pack_len}')

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = idx * self.pack_len
        e = s + self.pack_len
        return (torch.tensor(self.ids[s:e], dtype=torch.long),
                torch.tensor(self.labels[s:e], dtype=torch.long))


def make_collate_fn(max_seq_len):
    def collate(batch):
        input_ids = torch.stack([b[0] for b in batch])
        labels = torch.stack([b[1] for b in batch])
        return input_ids, labels
    return collate
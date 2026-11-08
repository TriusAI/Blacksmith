import pathlib
import argparse
import torch
import torch.nn.functional as F
from model import NanoGPT
from tokenmaker import Tokenizer, TokenizerNoIM, TokenizerVocabSize
from hyperparameters import d_model, n_heads, n_layers, max_seq_len

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=None,
                    help='checkpoint path (default: auto-detect SFT then pretrained)')
parser.add_argument('--rope-scale', type=float, default=1.0,
                    help='YaRN context stretch factor (e.g. 2.0 to stretch 64k->128k)')
parser.add_argument('--max-seq-len', type=int, default=None,
                    help='inference context length (default: max_seq_len * rope_scale)')
parser.add_argument('--eot-between-turns', action=argparse.BooleanOptionalAction, default=True,
                    help='insert EOT before each user turn, i.e. each input starts a new '
                         'conversation (matches the EOT-separated UltraChat pretokenization). '
                         'disable with --no-eot-between-turns for one continuous conversation.')
parser.add_argument('--think', action='store_true',
                    help='seed the assistant turn with </think> to elicit reasoning '
                         '(matches the OpenThoughts pretokenization; off by default).')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

vocab_size = TokenizerVocabSize

# YaRN context extension: the model was TRAINED at max_seq_len (original_max_pos
# in the model); rope_scale > 1 stretches positions beyond that. The RoPE
# tables are built for the extended length so the KV cache can grow past
# the training context.
original_max_pos = max_seq_len
if args.max_seq_len is not None:
    inference_max_seq_len = args.max_seq_len
else:
    inference_max_seq_len = max_seq_len * args.rope_scale if args.rope_scale > 1.0 else max_seq_len

model = NanoGPT(vocab_size, d_model, n_heads, n_layers, inference_max_seq_len,
                rope_scale=args.rope_scale,
                original_max_pos=original_max_pos).to(device)
model = model.to(torch.bfloat16)

if args.ckpt is not None:
    chkp_path = args.ckpt
elif pathlib.Path('blacksmith-1_sft.pt').exists():
    chkp_path = 'blacksmith-1_sft.pt'
else:
    chkp_path = 'blacksmith-1_model.pt'
chkp = torch.load(chkp_path, map_location=device)
print(f'loaded {chkp_path}')
model.load_state_dict(chkp['model'])
model.eval()

END_OF_TEXT = Tokenizer.encode('<|endoftext|>')[0]
IM_END = Tokenizer.encode('<|im_end|>')[0]
IM_START = Tokenizer.encode('<|im_start|>')[0]
IM_ROLE_USER = Tokenizer.encode('user')
IM_ROLE_ASSISTANT = Tokenizer.encode('assistant')
NEWLINE = Tokenizer.encode('\n')[0]
EOT = Tokenizer.eos_token_id  # 50256; conversation separator in the .bin corpora
THINK_START = Tokenizer.encode('<think>')[0]
THINK_END = Tokenizer.encode('</think>')[0]


@torch.no_grad()
def gen_step(model, token_ids, k_caches, v_caches, temperature, pos_offset,
             top_k, repetition_penalty, recent_tokens):
    logits, k_caches, v_caches = model(
        token_ids, k_cache=k_caches, v_cache=v_caches,
        infer=True, pos_offset=pos_offset,
    )
    next_logits = logits[0, -1, :] / temperature

    if repetition_penalty != 1.0 and recent_tokens.numel() > 0:
        gathered = next_logits.gather(0, recent_tokens)
        gathered = torch.where(
            gathered < 0, gathered * repetition_penalty,
            gathered / repetition_penalty,
        )
        next_logits.scatter_(0, recent_tokens, gathered)

    if top_k > 0:
        topk_vals = torch.topk(next_logits, top_k).values
        threshold = topk_vals[-1]
        next_logits = torch.where(
            next_logits < threshold,
            torch.tensor(float('-inf'), device=device), next_logits,
        )
    probs = F.softmax(next_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token, k_caches, v_caches


def generate(model, token_ids, n_new_tokens, temperature=0.7, top_k=25,
             repetition_penalty=1.3, repeat_window=64, stop_tokens=None,
             print_fn=None):
    token_ids = token_ids.to(device)
    recent = torch.tensor([], dtype=torch.long, device=device)
    if stop_tokens is None:
        stop_tokens = (END_OF_TEXT, IM_END)
    if print_fn is None:
        print_fn = lambda tok: None

    next_token, k_caches, v_caches = gen_step(
        model, token_ids, None, None, temperature, 0, top_k,
        repetition_penalty, recent,
    )
    if next_token.item() in stop_tokens:
        return []
    generated = [next_token.item()]
    print_fn(next_token.item())
    pos = token_ids.shape[1]
    token_ids = torch.cat([token_ids, next_token[None, :]], dim=1)
    recent = torch.cat([recent, next_token])

    for _ in range(n_new_tokens):
        window = recent[-repeat_window:]
        next_token, k_caches, v_caches = gen_step(
            model, next_token[None, :], k_caches, v_caches,
            temperature, pos, top_k, repetition_penalty, window,
        )
        if next_token.item() in stop_tokens:
            break
        generated.append(next_token.item())
        print_fn(next_token.item())
        token_ids = torch.cat([token_ids, next_token[None, :]], dim=1)
        recent = torch.cat([recent, next_token])
        if k_caches[0].shape[2] >= inference_max_seq_len:
            k_caches = [k[:, :, -inference_max_seq_len:, :] for k in k_caches]
            v_caches = [v[:, :, -inference_max_seq_len:, :] for v in v_caches]
        pos += 1
    return generated


context = []
while True:
    try:
        user_input = input('>>> ').strip()
    except (EOFError, KeyboardInterrupt):
        print('')
        break
    if user_input == '/quit':
        break
    if user_input == '/reset':
        context = []
        print('(context reset)')
        continue

    if args.eot_between_turns:
        context += [EOT]
    user_tokens = TokenizerNoIM.encode(user_input)
    context += [IM_START] + IM_ROLE_USER + [NEWLINE]
    context += user_tokens
    context += [IM_END, NEWLINE]
    context += [IM_START] + IM_ROLE_ASSISTANT + [NEWLINE]
    if args.think:
        # OpenThoughts format: assistant turn is
        #   <thinking>{reasoning}</thinking>\n{answer}<|im_end|>
        context += [THINK_START]

    if len(context) > inference_max_seq_len:
        # Truncate from the front, but only at a boundary so we don't split
        # a turn: prefer an EOT boundary (EOT followed by <|im_start|>),
        # falling back to a turn boundary (<|im_end|> NEWLINE <|im_start|>).
        cut = len(context) - inference_max_seq_len
        for i in range(cut, len(context)):
            if i >= 1 and context[i-1] == EOT and context[i] == IM_START:
                context = context[i:]
                break
            if (i >= 2 and context[i-1] == NEWLINE and context[i-2] == IM_END
                    and context[i] == IM_START):
                context = context[i:]
                break
        else:
            context = context[-inference_max_seq_len:]

    tensor = torch.tensor([context]).to(device)
    # Phase 1: reasoning (think mode only). The assistant turn was seeded
    # with </think> (OpenThoughts format); generation stops at </think>,
    # or at EOT/<|im_end|> if the model never closes it. In either case we
    # append the </think>\n boundary manually so the turn stays structurally
    # complete and the answer phase sees the same context the corpus had.
    answer_seed = [NEWLINE]
    if args.think:
        def print_think_token(tok):
            if tok != THINK_END:
                print(Tokenizer.decode([tok]), end='', flush=True)
        think_ids = generate(
            model, tensor, 1024,
            temperature=0.7, top_k=25,
            repetition_penalty=1.3, repeat_window=64,
            stop_tokens=(THINK_END, END_OF_TEXT, IM_END),
            print_fn=print_think_token,
        )
        print('', flush=True)  # end of the thinking block
        context += think_ids
        context += [THINK_END, NEWLINE]
        answer_seed = []

    # Phase 2: final answer. In think mode the model resumes right after
    # </think>\n and produces the answer; in non-think mode answer_seed
    # supplies the NEWLINE that ends the `assistant\n` prefix.
    tensor = torch.tensor([context + answer_seed]).to(device)
    if args.think:
        print('---', flush=True)
    response_ids = generate(
        model, tensor, 500,
        temperature=0.7, top_k=25,
        repetition_penalty=1.3, repeat_window=64,
        stop_tokens=(END_OF_TEXT, IM_END),
    )
    response_text = Tokenizer.decode(response_ids)
    print(response_text, flush=True)

    context += answer_seed + response_ids
    context += [IM_END, NEWLINE]

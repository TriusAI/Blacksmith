from transformers import GPT2Tokenizer

# we use the format `<|im_start|>{role}\n{content}<|im_end|>`, e.g.:
# 
#     User: What color is the sky?
#     Assistant: Blue, when it's sunny and the sun is not blocked
#                by the clouds.
# 
# would be encoded as:
# 
#     <|im_start|>user
#     What color is the sky?<|im_end|>
#     <|im_start|>assistant
#     Blue, when it's sunny and the sun is not blocked by the clouds.<|im_end|>

TokenizerNoIM = GPT2Tokenizer.from_pretrained('gpt2')

TokenizerNoThinking = GPT2Tokenizer.from_pretrained('gpt2')
_additional_token = [
    '<|im_start|>',
    '<|im_end|>',
]
TokenizerNoThinking.add_special_tokens({'additional_special_tokens': _additional_token})

Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
_additional_token = [
    '<|im_start|>',
    '<|im_end|>',
    '<think>',
    '</think>',
]
Tokenizer.add_special_tokens({'additional_special_tokens': _additional_token})

TokenizerVocabSize = Tokenizer.vocab_size + len(_additional_token)


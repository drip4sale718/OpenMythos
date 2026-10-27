"""
Transcribe audio or video with Whisper and feed it to OpenMythos.

    pip install openai-whisper
    python examples/speech_example.py talk.mp4 [more files ...]
"""

import sys

from torch.utils.data import DataLoader

from open_mythos import (
    MythosConfig,
    MythosTokenizer,
    OpenMythos,
    TranscriptDataset,
    WhisperTranscriber,
)

paths = sys.argv[1:]
if not paths:
    sys.exit("usage: python examples/speech_example.py <audio-or-video> [...]")

tokenizer = MythosTokenizer()
transcriber = WhisperTranscriber("base")

text = transcriber(paths[0])
print(f"Transcript of {paths[0]}:\n{text}\n")

cfg = MythosConfig(
    vocab_size=tokenizer.vocab_size,
    dim=256,
    n_heads=8,
    n_kv_heads=2,
    max_seq_len=128,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    n_experts=8,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=64,
    lora_rank=8,
    attn_type="gqa",
)
model = OpenMythos(cfg)

ds = TranscriptDataset(paths, tokenizer, seq_len=64, transcriber=transcriber)
loader = DataLoader(ds, batch_size=2)

for x, y in loader:
    logits = model(x, n_loops=4)
    print(f"Batch {tuple(x.shape)} → logits {tuple(logits.shape)}")
    break
else:
    print("Transcripts were too short to fill one 64-token sequence.")

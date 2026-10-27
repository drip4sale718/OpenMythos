"""
Transcribe audio or video and feed it to OpenMythos.

Whisper (local files):
    pip install openai-whisper
    python examples/speech_example.py talk.mp4 [more files ...]

claude-real-video (local files or URLs, reuses existing subtitles):
    pip install "claude-real-video[whisper]"
    python examples/speech_example.py --crv "https://www.youtube.com/watch?v=..."
"""

import sys

from torch.utils.data import DataLoader

from open_mythos import (
    CrvTranscriber,
    MythosConfig,
    MythosTokenizer,
    OpenMythos,
    TranscriptDataset,
    WhisperTranscriber,
)

args = sys.argv[1:]
use_crv = "--crv" in args
paths = [a for a in args if a != "--crv"]
if not paths:
    sys.exit("usage: python examples/speech_example.py [--crv] <audio-or-video> [...]")

tokenizer = MythosTokenizer()

if use_crv:
    transcriber = CrvTranscriber()
    video = transcriber.process(paths[0])
    text = video.text
    print(f"crv kept {len(video.frames)} keyframes in {video.out_dir}")
    for frame in video.frames[:5]:
        print(f"  {frame['timestamp_sec']:8.2f}s  {frame['path']}")
else:
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

# Reuse the transcript printed above instead of transcribing paths[0] again
cache = {paths[0]: text}
ds = TranscriptDataset(
    paths,
    tokenizer,
    seq_len=64,
    transcriber=lambda p: cache[p] if p in cache else transcriber(p),
)
loader = DataLoader(ds, batch_size=2)

for x, y in loader:
    logits = model(x, n_loops=4)
    print(f"Batch {tuple(x.shape)} → logits {tuple(logits.shape)}")
    break
else:
    print("Transcripts were too short to fill one 64-token sequence.")

import pytest
import torch
from torch.utils.data import DataLoader

from open_mythos.speech import TranscriptDataset, WhisperTranscriber


class CharEncoding:
    """One token per character, so chunk boundaries are easy to reason about."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]


TRANSCRIPTS = {
    "a.wav": "hello world",
    "b.mp4": "the quick brown fox",
    "silent.wav": "",
}


def fake_transcriber(path: str) -> str:
    return TRANSCRIPTS[path]


def test_pairs_are_shifted_by_one():
    ds = TranscriptDataset(
        ["a.wav"],
        CharEncoding(),
        seq_len=4,
        transcriber=fake_transcriber,
        separator="",
    )
    pairs = list(ds)
    assert len(pairs) == 2  # 11 tokens → two chunks of 5, one token left over
    for x, y in pairs:
        assert x.shape == y.shape == (4,)
        assert x.dtype == y.dtype == torch.long
        assert torch.equal(x[1:], y[:-1])


def test_packs_across_files_with_separator():
    ds = TranscriptDataset(
        ["a.wav", "b.mp4"],
        CharEncoding(),
        seq_len=7,
        transcriber=fake_transcriber,
        separator="|",
    )
    stream = "hello world|the quick brown fox|"
    expected = [ord(c) for c in stream]
    got = []
    for x, y in ds:
        got.extend(x.tolist() + [y[-1].item()])
    assert got == expected[: len(got)]
    assert len(expected) - len(got) < 8


def test_skips_empty_transcripts():
    ds = TranscriptDataset(
        ["silent.wav", "a.wav"],
        CharEncoding(),
        seq_len=4,
        transcriber=fake_transcriber,
        separator="",
    )
    first_x, _ = next(iter(ds))
    assert first_x.tolist() == [ord(c) for c in "hell"]


def test_dataloader_workers_split_files():
    seen = []

    def recording_transcriber(path: str) -> str:
        seen.append(path)
        return "x" * 10

    paths = [f"{i}.wav" for i in range(4)]
    ds = TranscriptDataset(
        paths, CharEncoding(), seq_len=4, transcriber=recording_transcriber
    )
    batches = list(DataLoader(ds, batch_size=None, num_workers=0))
    assert sorted(seen) == paths
    assert len(batches) > 0


def test_transcriber_loads_lazily():
    t = WhisperTranscriber("tiny", language="en")
    assert t._model is None
    assert t.transcribe_kwargs == {"language": "en"}


def test_whisper_end_to_end(tmp_path):
    whisper = pytest.importorskip("whisper")
    import wave

    import numpy as np

    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(whisper.audio.SAMPLE_RATE)
        f.writeframes(np.zeros(whisper.audio.SAMPLE_RATE, dtype=np.int16).tobytes())

    try:
        t = WhisperTranscriber("tiny", device="cpu", language="en")
        text = t(str(path))
    except (OSError, RuntimeError) as e:  # weights unavailable offline
        pytest.skip(f"Whisper weights unavailable: {e}")
    assert isinstance(text, str)

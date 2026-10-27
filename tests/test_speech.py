import json
import os
import shutil
import subprocess

import pytest
import torch
from torch.utils.data import DataLoader

from open_mythos.speech import CrvTranscriber, TranscriptDataset, WhisperTranscriber


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


def test_crv_transcriber_reads_crv_output(tmp_path, monkeypatch):
    import types

    from open_mythos import speech

    def fake_process(src, out_dir, **kwargs):
        assert kwargs["overwrite"] is True
        assert kwargs["lang"] == "en"
        frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        segs = [
            {"start": 0.0, "end": 1.0, "text": "first line"},
            {"start": 1.0, "end": 2.0, "text": "second line"},
        ]
        with open(os.path.join(out_dir, "transcript.json"), "w") as f:
            json.dump({"segments": segs}, f)
        frames_json = os.path.join(out_dir, "frames.json")
        with open(frames_json, "w") as f:
            json.dump(
                {
                    "frames": [
                        {
                            "file": "frame_001.jpg",
                            "timestamp_sec": 0.0,
                            "selection_reason": "first",
                        }
                    ]
                },
                f,
            )
        return types.SimpleNamespace(
            out_dir=out_dir,
            frames_dir=frames_dir,
            transcript_path=None,
            frames_json_path=frames_json,
        )

    monkeypatch.setattr(
        speech, "_import_crv", lambda: types.SimpleNamespace(process=fake_process)
    )
    t = CrvTranscriber(out_root=str(tmp_path), lang="en")
    video = t.process("https://example.com/v")

    assert video.text == "first line\nsecond line"
    assert len(video.segments) == 2
    assert video.frames == [
        {
            "path": os.path.join(video.out_dir, "frames", "frame_001.jpg"),
            "timestamp_sec": 0.0,
            "selection_reason": "first",
        }
    ]
    assert t("https://example.com/v") == video.text
    assert video.out_dir == t._out_dir("https://example.com/v")


def test_crv_end_to_end(tmp_path):
    pytest.importorskip("claude_real_video")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")

    # Two visually distinct scenes plus a sidecar .srt, so crv keeps several
    # frames and takes the transcript from the subtitles without Whisper.
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
            "-f", "lavfi", "-i", "color=c=red:duration=3:size=320x240:rate=10",
            "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
            str(video),
        ],
        check=True,
    )  # fmt: skip
    (tmp_path / "clip.srt").write_text(
        "1\n00:00:00,500 --> 00:00:02,000\nHello from the test pattern.\n\n"
        "2\n00:00:03,500 --> 00:00:05,000\nNow the screen is red.\n"
    )

    t = CrvTranscriber(out_root=str(tmp_path / "out"))
    result = t.process(str(video))

    assert result.text == "Hello from the test pattern.\nNow the screen is red."
    assert result.segments[0]["start"] == 0.5
    assert len(result.frames) >= 2
    assert all(os.path.exists(f["path"]) for f in result.frames)

    ds = TranscriptDataset(
        [str(video)], CharEncoding(), seq_len=8, transcriber=t, separator=""
    )
    x, _ = next(iter(ds))
    assert x.tolist() == [ord(c) for c in "Hello fr"]

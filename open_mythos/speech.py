"""
Speech-to-text front end for OpenMythos.

OpenMythos is a text-only model, so audio and video reach it as transcripts.
This module wraps OpenAI Whisper (https://github.com/openai/whisper) to turn
media files into text, then packs that text into fixed-length
`(input, target)` pairs in the same way as the FineWeb-Edu pretraining loader.

Whisper is an optional dependency and is imported lazily:

    pip install openai-whisper
    # or a fork, e.g.
    pip install git+https://github.com/drip4sale718/whisper_.git

Whisper decodes media through the `ffmpeg` binary, so any container ffmpeg
can read (wav, mp3, flac, mp4, mkv, webm, ...) works as input.

`CrvTranscriber` is an alternative front end built on claude-real-video
(https://github.com/HUANGCHIHHUNGLeo/claude-real-video). It also accepts video
URLs, reuses a video's own subtitles when it has them instead of running
Whisper, and keeps timestamped keyframes next to the transcript:

    pip install "claude-real-video[whisper]"
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import torch
from torch.utils.data import IterableDataset, get_worker_info

DEFAULT_WHISPER_MODEL = "base"


def _import_whisper():
    """Import Whisper, raising an actionable error when it is missing."""
    try:
        import whisper
    except ImportError as e:
        raise ImportError(
            "open_mythos.speech requires Whisper. Install it with "
            "`pip install openai-whisper` (and make sure `ffmpeg` is on PATH)."
        ) from e
    return whisper


class WhisperTranscriber:
    """
    Callable that transcribes an audio or video file to text with Whisper.

    The Whisper model is loaded on first use, so constructing a transcriber is
    cheap and it can be handed to DataLoader workers before any weights exist.

    Args:
        model_name (str): Whisper checkpoint name ("tiny", "base", "small",
            "medium", "large-v3", "turbo", ...) or a path to a local checkpoint.
        device (str | None): Torch device for Whisper. Defaults to CUDA when
            available, otherwise CPU.
        **transcribe_kwargs: Forwarded to `whisper.Whisper.transcribe`, e.g.
            `language="en"` or `task="translate"`.

    Example:
        >>> transcribe = WhisperTranscriber("tiny")
        >>> text = transcribe("talk.mp4")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_WHISPER_MODEL,
        device: str | None = None,
        **transcribe_kwargs,
    ):
        self.model_name = model_name
        self.device = device
        self.transcribe_kwargs = transcribe_kwargs
        self._model = None

    @property
    def model(self):
        """The loaded Whisper model (loaded lazily on first access)."""
        if self._model is None:
            whisper = _import_whisper()
            self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def transcribe(self, path: str) -> dict:
        """
        Run Whisper on a media file and return its full result.

        Args:
            path (str): Path to an audio or video file.

        Returns:
            dict: Whisper's result with "text", "segments" and "language".
        """
        kwargs = dict(self.transcribe_kwargs)
        # fp16 is only supported on CUDA; silence Whisper's CPU warning
        kwargs.setdefault("fp16", self.model.device.type == "cuda")
        return self.model.transcribe(path, **kwargs)

    def __call__(self, path: str) -> str:
        """
        Transcribe a media file to plain text.

        Args:
            path (str): Path to an audio or video file.

        Returns:
            str: The transcript with surrounding whitespace stripped.
        """
        return self.transcribe(path)["text"].strip()


def _import_crv():
    """Import claude-real-video, raising an actionable error when it is missing."""
    try:
        from claude_real_video import core
    except ImportError as e:
        raise ImportError(
            "CrvTranscriber requires claude-real-video. Install it with "
            '`pip install "claude-real-video[whisper]"` (and make sure `ffmpeg` '
            "is on PATH)."
        ) from e
    return core


@dataclass
class VideoTranscript:
    """
    Everything claude-real-video extracted from one video.

    Attributes:
        text (str): The transcript, one segment per line.
        segments (list[dict]): Timestamped segments as
            `{"start": float, "end": float, "text": str}`, in seconds.
        frames (list[dict]): Kept keyframes as
            `{"path": str, "timestamp_sec": float, "selection_reason": str}`.
        out_dir (str): Directory holding crv's raw output for this video.
    """

    text: str
    segments: list[dict] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)
    out_dir: str = ""


class CrvTranscriber:
    """
    Callable that turns a video file or URL into text with claude-real-video.

    crv downloads URLs with yt-dlp, uses the video's own subtitles (a sidecar
    `.srt`/`.vtt`, an embedded track or platform captions) when they exist and
    falls back to Whisper otherwise. It also keeps one frame per scene change,
    which `process` returns with source timestamps. OpenMythos only reads the
    text; the frames are kept for inspection or a future vision encoder.

    Args:
        out_root (str | None): Directory for crv's output, one subdirectory per
            source. Defaults to a fresh temporary directory.
        whisper_model (str): Whisper checkpoint used when no subtitles exist.
        lang (str): Transcript language, e.g. "en", or "auto" to detect it.
        **process_kwargs: Forwarded to `claude_real_video.core.process`, e.g.
            `start="1:00"`, `end="5:00"` or `adaptive=True`.

    Example:
        >>> transcribe = CrvTranscriber()
        >>> text = transcribe("https://www.youtube.com/watch?v=...")
    """

    def __init__(
        self,
        out_root: str | None = None,
        whisper_model: str = DEFAULT_WHISPER_MODEL,
        lang: str = "auto",
        **process_kwargs,
    ):
        self.out_root = out_root or tempfile.mkdtemp(prefix="open-mythos-crv-")
        self.whisper_model = whisper_model
        self.lang = lang
        self.process_kwargs = process_kwargs

    def _out_dir(self, src: str) -> str:
        """Stable per-source output directory, so reruns replace old output."""
        digest = hashlib.sha1(src.encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.out_root, digest)

    def process(self, src: str) -> VideoTranscript:
        """
        Run claude-real-video on a video and collect its transcript and frames.

        Args:
            src (str): Local video path or URL (YouTube, Instagram, ...).

        Returns:
            VideoTranscript: Transcript text, timed segments and keyframes.
        """
        core = _import_crv()
        out_dir = self._out_dir(src)
        result = core.process(
            src,
            out_dir,
            overwrite=True,
            whisper_model=self.whisper_model,
            lang=self.lang,
            **self.process_kwargs,
        )

        segments = []
        segments_path = os.path.join(result.out_dir, "transcript.json")
        if os.path.exists(segments_path):
            with open(segments_path, encoding="utf-8") as f:
                segments = json.load(f).get("segments", [])

        if segments:
            text = "\n".join(s["text"] for s in segments)
        elif result.transcript_path and os.path.exists(result.transcript_path):
            with open(result.transcript_path, encoding="utf-8") as f:
                text = f.read()
        else:
            text = ""

        frames = []
        if result.frames_json_path and os.path.exists(result.frames_json_path):
            with open(result.frames_json_path, encoding="utf-8") as f:
                for fr in json.load(f).get("frames", []):
                    frames.append(
                        {
                            "path": os.path.join(result.frames_dir, fr["file"]),
                            "timestamp_sec": fr["timestamp_sec"],
                            "selection_reason": fr.get("selection_reason", ""),
                        }
                    )

        return VideoTranscript(
            text=text.strip(), segments=segments, frames=frames, out_dir=out_dir
        )

    def __call__(self, src: str) -> str:
        """
        Transcribe a video file or URL to plain text.

        Args:
            src (str): Local video path or URL.

        Returns:
            str: The transcript with surrounding whitespace stripped.
        """
        return self.process(src).text


class TranscriptDataset(IterableDataset):
    """
    Transcribe media files and yield fixed-length (input, target) pairs.

    Each file is transcribed, tokenized and appended to a rolling buffer that
    is sliced into `seq_len + 1` chunks, exactly like `FineWebEduDataset` in
    `training/3b_fine_web_edu.py`. Unlike that loader this one is finite: it
    makes a single pass over `paths` and drops the final partial chunk.

    Files are split across DataLoader workers round-robin, so each file is
    transcribed by exactly one worker.

    Args:
        paths (Iterable[str]): Audio or video files to transcribe.
        encoding: Tokenizer exposing `.encode(str) -> list[int]`, e.g.
            `MythosTokenizer`.
        seq_len (int): Context length; every yielded pair has this many tokens.
        transcriber (Callable[[str], str] | None): Maps a file path to its
            transcript. Defaults to `WhisperTranscriber()`; pass
            `CrvTranscriber()` to accept video URLs as `paths`.
        separator (str): Text placed between consecutive transcripts so the
            model sees a boundary between unrelated recordings.

    Example:
        >>> ds = TranscriptDataset(["a.mp3", "b.mp4"], MythosTokenizer(), 512)
        >>> x, y = next(iter(ds))
    """

    def __init__(
        self,
        paths: Iterable[str],
        encoding,
        seq_len: int,
        transcriber: Callable[[str], str] | None = None,
        separator: str = "\n\n",
    ):
        self.paths = list(paths)
        self.encoding = encoding
        self.seq_len = seq_len
        self.transcriber = transcriber or WhisperTranscriber()
        self.separator = separator

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Yield `(input_ids, target_ids)` tensors of length `seq_len`.

        `target[i] == input[i + 1]` for next-token prediction.
        """
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0

        buf = []
        for path in self.paths[worker_id::num_workers]:
            text = self.transcriber(path)
            if not text:
                continue
            buf.extend(self.encoding.encode(text + self.separator))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )

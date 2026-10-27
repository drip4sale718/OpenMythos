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
"""

from collections.abc import Callable, Iterable, Iterator

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
            transcript. Defaults to `WhisperTranscriber()`.
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

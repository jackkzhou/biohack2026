"""ElevenLabs TTS that clones a voice from a local audio file and plays it locally.

Pipeline: ML predicted word -> cloned/selected ElevenLabs voice -> speaker output.
Unlike backend/tts.py (which streams to the browser from a pre-made voice_id),
this is a standalone script for testing the local-file voice-cloning path on a
Mac: it calls client.voices.add() once per reference file (cached after that)
and plays the result directly through sounddevice.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
from elevenlabs import ElevenLabs
from pydub import AudioSegment

log = logging.getLogger("tts_from_file_voice")

MODEL_ID = "eleven_turbo_v2_5"
OUTPUT_FORMAT = "mp3_44100_128"

_client: ElevenLabs | None = None
_voice_cache: dict[str, str] = {}  # reference file path -> cloned voice_id


def _get_client() -> ElevenLabs:
    global _client
    if _client is None:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set in the environment")
        _client = ElevenLabs(api_key=api_key)
    return _client


def clone_voice_from_file(file_path: str, voice_name: str = "cloned_user_voice") -> str:
    """Instant-clone a voice from a local audio sample, returning its voice_id.

    Cached by file path for the lifetime of the process so repeated calls
    (e.g. one per predicted word) don't re-upload the sample every time.
    """
    cached = _voice_cache.get(file_path)
    if cached:
        return cached

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Reference audio file not found: {path}")

    client = _get_client()
    with open(path, "rb") as f:
        # elevenlabs>=1.0 moved instant voice cloning under voices.ivc; older SDKs have voices.add.
        if hasattr(client.voices, "ivc"):
            voice = client.voices.ivc.create(name=voice_name, files=[f])
        else:
            voice = client.voices.add(name=voice_name, files=[f])

    _voice_cache[file_path] = voice.voice_id
    log.info("Cloned voice %r from %s -> voice_id=%s", voice_name, path, voice.voice_id)
    return voice.voice_id


def resolve_voice_id(voice_id_or_file_path: str) -> str:
    """Existing voice_id if given, otherwise clone (or reuse a cached clone) from a file."""
    if os.path.isfile(voice_id_or_file_path):
        return clone_voice_from_file(voice_id_or_file_path)
    return voice_id_or_file_path


def _resolve_output_device(output_device_index: int | str | None) -> int | None:
    """None -> system default. int -> used as-is. str -> substring match on device name."""
    if output_device_index is None or isinstance(output_device_index, int):
        return output_device_index
    needle = output_device_index.lower()
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0 and needle in dev["name"].lower():
            return idx
    raise ValueError(f"No output device matching {output_device_index!r} found")


def speak_ml_prediction(
    predicted_word: str,
    voice_id_or_file_path: str,
    output_device_index: int | str | None = None,
) -> None:
    """Synthesize `predicted_word` in the target voice and play it out loud.

    `voice_id_or_file_path` is either an existing ElevenLabs voice_id, or the
    path to a local reference audio file to clone from.
    """
    client = _get_client()
    voice_id = resolve_voice_id(voice_id_or_file_path)

    audio_chunks = client.text_to_speech.convert(
        voice_id=voice_id,
        model_id=MODEL_ID,
        output_format=OUTPUT_FORMAT,
        text=predicted_word,
    )
    mp3_bytes = b"".join(audio_chunks)

    # Written to a temp file rather than decoded from BytesIO: some ffmpeg
    # builds can't seek a piped stdin stream and fail to decode in-memory MP3s.
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        tmp.write(mp3_bytes)
        tmp.flush()
        segment = AudioSegment.from_file(tmp.name, format="mp3")

    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    if segment.channels > 1:
        samples = samples.reshape((-1, segment.channels))
    samples /= float(1 << (8 * segment.sample_width - 1))  # normalize PCM ints to [-1, 1]

    device = _resolve_output_device(output_device_index)
    sd.play(samples, samplerate=segment.frame_rate, device=device)
    sd.wait()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Test: ML predicted text -> local-file voice clone -> speaker output")
    parser.add_argument("word", nargs="?", default="hello world", help="Predicted word/phrase from the ML classifier")
    parser.add_argument(
        "voice_file",
        nargs="?",
        default="user_voice_sample.wav",
        help="Local reference audio file to clone (or an existing voice_id)",
    )
    parser.add_argument("--device", default=None, help="Output device index or name substring (default: system speakers)")
    args = parser.parse_args()

    device: int | str | None = args.device
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            pass  # treat as a name substring

    speak_ml_prediction(args.word, args.voice_file, device)

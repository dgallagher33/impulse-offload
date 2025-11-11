"""Tests for the Whisper integration helpers."""
from __future__ import annotations

import wave
from array import array
from pathlib import Path

import pytest

from impulse_offloader.utils import whisper_integration


class SimpleArray:
    """Lightweight array structure supporting minimal numpy-like operations."""

    def __init__(self, values):
        self.values = values

    @property
    def ndim(self) -> int:
        if not self.values:
            return 1
        return 2 if isinstance(self.values[0], list) else 1

    def astype(self, dtype: str) -> "SimpleArray":
        if dtype == SimpleNumpy.float32:
            return SimpleArray(SimpleNumpy._to_float(self.values))
        if dtype == SimpleNumpy.int16:
            return SimpleArray(SimpleNumpy._to_int(self.values))
        raise ValueError(f"Unsupported dtype: {dtype}")

    def __mul__(self, other: float) -> "SimpleArray":
        return SimpleArray(SimpleNumpy._apply_scalar(self.values, float(other)))

    def __truediv__(self, other: float) -> "SimpleArray":
        return SimpleArray(SimpleNumpy._apply_scalar(self.values, 1 / float(other)))

    def tobytes(self) -> bytes:
        flat = SimpleNumpy._flatten(self.values)
        return array("h", [int(round(x)) for x in flat]).tobytes()


class SimpleNumpy:
    """Minimal numpy replacement used for unit testing without the dependency."""

    float32 = "float32"
    int16 = "int16"

    @staticmethod
    def _flatten(values):
        if not values:
            return []
        if isinstance(values[0], list):
            return [item for row in values for item in row]
        return list(values)

    @staticmethod
    def _to_float(values):
        if isinstance(values, SimpleArray):
            values = values.values
        if not values:
            return []
        if isinstance(values[0], list):
            return [[float(x) for x in row] for row in values]
        return [float(x) for x in values]

    @staticmethod
    def _to_int(values):
        floats = SimpleNumpy._to_float(values)
        if not floats:
            return []
        if isinstance(floats[0], list):
            return [[int(round(x)) for x in row] for row in floats]
        return [int(round(x)) for x in floats]

    @staticmethod
    def _apply_scalar(values, scalar: float):
        base = SimpleNumpy._to_float(values)
        if not base:
            return []
        if isinstance(base[0], list):
            return [[x * scalar for x in row] for row in base]
        return [x * scalar for x in base]

    def asarray(self, data, dtype=None):
        if isinstance(data, SimpleArray):
            array_data = data.values
        else:
            array_data = data
        if dtype == self.float32:
            array_data = self._to_float(array_data)
        return SimpleArray(array_data)

    def squeeze(self, data):
        if isinstance(data, SimpleArray):
            values = data.values
        else:
            values = data
        if not values:
            return SimpleArray([])
        if isinstance(values[0], list) and len(values[0]) == 1:
            squeezed = [row[0] for row in values]
            return SimpleArray(squeezed)
        return SimpleArray(values)

    def expand_dims(self, data, axis):
        if axis != 0:
            raise ValueError("Only axis=0 is supported in tests")
        if isinstance(data, SimpleArray):
            values = data.values
        else:
            values = data
        expanded = [[value] for value in values]
        return SimpleArray(expanded)

    def abs(self, data):
        values = self._to_float(data)
        if isinstance(values, list) and values and isinstance(values[0], list):
            return SimpleArray([[abs(x) for x in row] for row in values])
        return SimpleArray([abs(x) for x in values])

    def max(self, data):
        values = self._to_float(data)
        if not values:
            return 0.0
        if isinstance(values[0], list):
            return max(max(row) for row in values)
        return max(values)

    def clip(self, data, lower: float, upper: float):
        values = self._to_float(data)
        if isinstance(values, list) and values and isinstance(values[0], list):
            clipped = [[min(max(x, lower), upper) for x in row] for row in values]
        else:
            clipped = [min(max(x, lower), upper) for x in values]
        return SimpleArray(clipped)

    class _IInfo:
        def __init__(self, max_value: int) -> None:
            self.max = max_value

    def iinfo(self, dtype: str) -> "SimpleNumpy._IInfo":
        if dtype != self.int16:
            raise ValueError("Only int16 is supported in tests")
        return SimpleNumpy._IInfo(32767)


class FakeSoundDevice:
    """Simple stub mimicking the parts of sounddevice used in tests."""

    devices = [
        {
            "name": "Fake Microphone",
            "index": 0,
            "max_input_channels": 1,
        }
    ]

    def query_devices(self, device=None, kind=None):  # noqa: D401 - simple stub
        if device is None:
            return self.devices
        return self.devices[0]

    def check_input_settings(self, **kwargs):  # pragma: no cover - no-op stub
        return None

    def rec(self, frames, samplerate, channels, dtype, device=None):
        values = [[(i / frames) * 0.5 for _ in range(channels)] for i in range(frames)]
        return values

    def wait(self):  # pragma: no cover - nothing to wait for in tests
        return None


class FakeWavfile:
    """Stub for scipy.io.wavfile."""

    def write(self, path: str, sample_rate: int, data) -> None:
        if isinstance(data, SimpleArray):
            values = data.values
            ndim = data.ndim
        else:
            values = data
            ndim = 2 if values and isinstance(values[0], list) else 1
        channels = 1 if ndim == 1 else len(values[0])
        payload = SimpleArray(values).astype(SimpleNumpy.int16).tobytes()
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(payload)


@pytest.fixture(autouse=True)
def _reset_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    simple_np = SimpleNumpy()
    monkeypatch.setattr(whisper_integration, "sounddevice", FakeSoundDevice())
    monkeypatch.setattr(whisper_integration, "wavfile", FakeWavfile())
    monkeypatch.setattr(whisper_integration, "_np", simple_np)


def test_record_audio_creates_wav_linux(tmp_path: Path) -> None:
    output_path = tmp_path / "output.wav"
    recorded = whisper_integration.record_audio(output_path, max_seconds=1)

    assert recorded == output_path
    assert output_path.exists()

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1


def test_transcribe_local_missing_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dummy_audio = tmp_path / "dummy.wav"
    dummy_audio.write_bytes(b"")

    monkeypatch.setattr(whisper_integration.shutil, "which", lambda _: None)

    with pytest.raises(whisper_integration.TranscriptionError) as exc:
        whisper_integration.transcribe_local(dummy_audio)

    assert "ffmpeg" in str(exc.value)


def test_capture_and_transcribe_local_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected_audio = tmp_path / "Inbox" / "audio"

    def fake_record(path: Path, **_: object) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")
        return path

    monkeypatch.setattr(whisper_integration, "record_audio", fake_record)
    monkeypatch.setattr(whisper_integration, "transcribe_local", lambda *_args, **_kwargs: "Hello")

    audio_path, transcription = whisper_integration.capture_and_transcribe_local(tmp_path)

    assert audio_path.parent == expected_audio
    assert audio_path.suffix == ".wav"
    assert transcription == "Hello"

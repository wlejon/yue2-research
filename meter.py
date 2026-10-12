"""Run loading and the audio/latent measurement kit.

A run directory (written by native/gen.js or lab.py) holds:
  plan.json, run.json, semantic.i32, latents.f32 ([frames, 64]),
  audio.f32 (planar stereo: [2, N] at 48 kHz), hidden_means.f32 + hidden_keys.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 48000
HOP = 1920          # samples per latent frame (25 Hz)
FPS = 25


@dataclass
class Run:
    dir: Path
    plan: dict
    info: dict
    semantic: np.ndarray | None
    latents: np.ndarray | None
    audio: np.ndarray | None

    @property
    def name(self):
        return self.dir.name


def load_run(d) -> Run:
    d = Path(d)
    def opt(name, dtype):
        p = d / name
        return np.fromfile(p, dtype=dtype) if p.exists() else None
    lat = opt("latents.f32", np.float32)
    aud = opt("audio.f32", np.float32)
    return Run(d, json.loads((d / "plan.json").read_text("utf-8")) if (d / "plan.json").exists() else {},
               json.loads((d / "run.json").read_text("utf-8")) if (d / "run.json").exists() else {},
               opt("semantic.i32", np.int32),
               None if lat is None else lat.reshape(-1, 64),
               None if aud is None else aud.reshape(2, -1))


def hidden_means(d):
    d = Path(d)
    keys = json.loads((d / "hidden_keys.json").read_text())
    m = np.fromfile(d / "hidden_means.f32", dtype=np.float32).reshape(len(keys), -1)
    return {(k["stage"], k["layer"], k["narEval"]): m[i] for i, k in enumerate(keys)}


def mono(audio):
    return audio.mean(axis=0) if audio.ndim == 2 else audio


# ── spectral front end, one column per latent frame (or finer) ──────────────
_MEL = {}


def logmel(x, sr=SR, hop=HOP // 4, n_mels=96, n_fft=4096):
    import librosa
    key = (sr, n_fft, n_mels)
    if key not in _MEL:
        _MEL[key] = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=30, fmax=16000)
    S = np.abs(librosa.stft(mono(x), n_fft=n_fft, hop_length=hop, center=True)) ** 2
    return np.log10(_MEL[key] @ S + 1e-8)          # [mels, cols]


def spectral_flux(lm):
    """Positive log-mel flux per column (column t vs t-1), mean over bands."""
    d = np.diff(lm, axis=1)
    return np.concatenate([[0.0], np.maximum(d, 0).mean(axis=0)])


# ── musical attribute meters (whole clip or a segment) ──────────────────────
def features(audio, sr=SR):
    """The attribute panel: loudness, dynamics, brightness, noisiness,
    rhythm density, tempo, percussive share, vocal-band share."""
    import librosa
    import pyloudnorm as pyln
    x = audio if audio.ndim == 2 else np.stack([audio, audio])
    m = mono(x)
    out = {}
    meter = pyln.Meter(sr)
    try:
        out["lufs"] = float(meter.integrated_loudness(x.T.astype(np.float64)))
    except Exception:
        out["lufs"] = float("nan")
    y = librosa.resample(m, orig_sr=sr, target_sr=22050)
    sr2 = 22050
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rdb = 20 * np.log10(rms + 1e-9)
    out["rms_db"] = float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))
    out["crest_db"] = float(np.percentile(rdb, 95) - np.percentile(rdb, 10))   # dynamics
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    out["centroid_hz"] = float(np.average(librosa.feature.spectral_centroid(S=S, sr=sr2)[0], weights=rms + 1e-9))
    out["flatness"] = float(np.average(librosa.feature.spectral_flatness(S=S)[0], weights=rms + 1e-9))
    out["rolloff_hz"] = float(np.average(librosa.feature.spectral_rolloff(S=S, sr=sr2)[0], weights=rms + 1e-9))
    env = librosa.onset.onset_strength(y=y, sr=sr2, hop_length=512)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr2, hop_length=512)
    out["onset_rate"] = float(len(onsets) / (len(y) / sr2))
    out["onset_strength"] = float(env.mean())
    tempo = librosa.feature.tempo(onset_envelope=env, sr=sr2, hop_length=512, aggregate=np.median)
    out["tempo"] = float(np.atleast_1d(tempo)[0])
    H, P = librosa.decompose.hpss(S)
    out["perc_ratio"] = float((P ** 2).sum() / ((H ** 2).sum() + (P ** 2).sum() + 1e-12))
    freqs = librosa.fft_frequencies(sr=sr2, n_fft=2048)
    band = (freqs > 250) & (freqs < 4000)
    out["vocal_band"] = float((H[band] ** 2).sum() / ((H ** 2).sum() + 1e-12))
    # low end share: bass / kick weight
    out["low_share"] = float((S[freqs < 150] ** 2).sum() / ((S ** 2).sum() + 1e-12))
    return out


FEATURE_KEYS = ["lufs", "rms_db", "crest_db", "centroid_hz", "flatness", "rolloff_hz", "onset_rate",
                "onset_strength", "tempo", "perc_ratio", "vocal_band", "low_share"]


def energy_score(f):
    """One scalar 'energy' summary used for ranking: loudness, brightness,
    rhythmic density and percussive share, each roughly unit-scaled."""
    return ((f["lufs"] + 14) / 4 + (f["centroid_hz"] - 2000) / 800 + (f["onset_rate"] - 3) / 1.5
            + (f["perc_ratio"] - 0.4) / 0.1) / 4


def segment(audio, t0, t1):
    a, b = int(t0 * SR), int(t1 * SR)
    return audio[..., a:b]

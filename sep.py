"""Exp 7: source separation of YuE2's output (htdemucs_ft: drums, bass,
other, vocals; MIT code and weights).

For each run dir with audio.f32 (planar stereo, 48 kHz): stems/<stem>.flac
(48 kHz stereo) and sep.json with the separation's GPU seconds per audio
second and each stem's energy relative to the mix (dB).

    .venv/Scripts/python sep.py runs/exp6 [runs/exp5/ref_folk_s11 ...] [--stems vocals]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SR = 48000


def run_dirs(paths):
    for p in paths:
        p = Path(p)
        if (p / "audio.f32").exists():
            yield p
        else:
            for d in sorted(p.rglob("audio.f32")):
                yield d.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--model", default="htdemucs_ft")
    ap.add_argument("--stems", default="vocals,drums,bass,other", help="stems written as FLAC")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    model = get_model(args.model).cuda().eval()
    msr = model.samplerate
    keep = set(args.stems.split(","))
    for d in run_dirs(args.paths):
        if (d / "sep.json").exists() and not args.force:
            continue
        a = np.fromfile(d / "audio.f32", dtype=np.float32).reshape(2, -1)
        secs = a.shape[1] / SR
        x = torch.from_numpy(a).cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        xi = AF.resample(x, SR, msr)
        ref = xi.mean(0)
        mu, sd = ref.mean(), ref.std() + 1e-8
        with torch.no_grad():
            out = apply_model(model, ((xi - mu) / sd)[None], device="cuda", shifts=0, split=True,
                              overlap=0.25, progress=False)[0] * sd + mu
        stems = {name: AF.resample(out[i], msr, SR)[:, :a.shape[1]] for i, name in enumerate(model.sources)}
        torch.cuda.synchronize()
        sep_s = time.perf_counter() - t0
        e_mix = float((x ** 2).mean())
        info = dict(model=args.model, seconds=sep_s, audio_seconds=secs, rtf=sep_s / secs, stems={})
        (d / "stems").mkdir(exist_ok=True)
        for name, s in stems.items():
            info["stems"][name] = dict(rel_db=float(10 * np.log10(float((s ** 2).mean()) / (e_mix + 1e-12) + 1e-12)))
            if name in keep:
                sf.write(d / "stems" / f"{name}.flac", s.cpu().numpy().T, SR, subtype="PCM_24")
            if name == "vocals":   # mono 16 kHz for native/stt.js
                AF.resample(s.mean(0), SR, 16000).cpu().numpy().astype(np.float32).tofile(d / "stems" / "vocals_16k.f32")
        (d / "sep.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(f"{d}: {secs:.0f} s audio, {sep_s:.2f} s ({sep_s / secs * 1000:.1f} ms per audio second); "
              + ", ".join(f"{k} {v['rel_db']:.1f} dB" for k, v in info["stems"].items()))


if __name__ == "__main__":
    main()

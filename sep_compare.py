"""Parity of brosoundml's native HTDemucs (brosoundml_separate) against sep.py's
oracle (demucs 4.0.1 htdemucs_ft through apply_model) on YuE2 output.

For each run dir with audio.f32 (planar stereo, 48 kHz), over the window
[--offset, --offset + --seconds) (whole song with --seconds 0), both sides
separate the same samples; per stem it reports the SDR of the native stem
against the oracle's,

    SDR = 10 log10( sum(ref^2) / sum((ref - est)^2) )   over both channels,

and each side's wall time. The oracle runs exactly as sep.py does (torchaudio
resample, mono-mix normalisation, apply_model split=True overlap 0.25 shifts=0)
with torch's defaults (cuDNN convolutions in TF32); --strict-fp32 turns TF32
off for a pure-FP32 reference. Oracle stems are cached under
<run>/sepcmp/ by window and mode.

    .venv/Scripts/python sep_compare.py runs/exp7/roll_adapt_w4 runs/exp7/scoref_jazz_s1 \
        --exe D:/projects/brosoundml/build/Release/brosoundml_separate.exe --seconds 10
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

SR = 48000
STEMS = ["drums", "bass", "other", "vocals"]


def oracle(a: np.ndarray, strict: bool):
    import torch
    import torchaudio.functional as AF
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    torch.backends.cudnn.allow_tf32 = not strict
    torch.backends.cuda.matmul.allow_tf32 = False
    model = get_model("htdemucs_ft").cuda().eval()
    msr = model.samplerate
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
    secs = time.perf_counter() - t0
    return {k: v.cpu().numpy().astype(np.float32) for k, v in stems.items()}, secs


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref = ref.astype(np.float64)
    err = ref - est.astype(np.float64)
    return float(10 * np.log10((ref ** 2).sum() / max((err ** 2).sum(), 1e-30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--exe", default="D:/projects/brosoundml/build/Release/brosoundml_separate.exe")
    ap.add_argument("--model", default="D:/projects/brosoundml/weights/htdemucs_ft")
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=10.0, help="0 = the whole song")
    ap.add_argument("--stems", default=",".join(STEMS))
    ap.add_argument("--strict-fp32", action="store_true")
    ap.add_argument("--json", help="write the table here")
    args = ap.parse_args()
    stems = args.stems.split(",")
    rows = []
    for run in args.runs:
        d = Path(run)
        full = np.fromfile(d / "audio.f32", dtype=np.float32).reshape(2, -1)
        s0 = int(args.offset * SR)
        n = full.shape[1] - s0 if args.seconds <= 0 else min(int(args.seconds * SR), full.shape[1] - s0)
        a = np.ascontiguousarray(full[:, s0:s0 + n])
        tag = f"o{args.offset:g}_s{args.seconds:g}_{'fp32' if args.strict_fp32 else 'tf32'}"
        cache = d / "sepcmp" / tag
        if all((cache / f"{s}.f32").exists() for s in STEMS):
            ref = {s: np.fromfile(cache / f"{s}.f32", dtype=np.float32).reshape(2, -1) for s in STEMS}
            osec = json.loads((cache / "oracle.json").read_text())["seconds"]
        else:
            ref, osec = oracle(a, args.strict_fp32)
            cache.mkdir(parents=True, exist_ok=True)
            for s, v in ref.items():
                v.tofile(cache / f"{s}.f32")
            (cache / "oracle.json").write_text(json.dumps({"seconds": osec}))
        with tempfile.TemporaryDirectory() as td:
            inp = Path(td) / "in.f32"
            a.tofile(inp)
            cmd = [args.exe, args.model, str(inp), str(Path(td) / "out"), "--rate", str(SR), "--stems", ",".join(stems)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise SystemExit(f"{d.name}: native run failed:\n{res.stdout}\n{res.stderr}")
            nat_ms = [float(l.split(":")[1].split("ms")[0]) for l in res.stdout.splitlines() if l.startswith("run ")]
            est = {s: np.fromfile(Path(td) / "out" / f"{s}.f32", dtype=np.float32).reshape(2, -1) for s in stems}
        row = {"run": d.name, "seconds": n / SR, "oracle_s": osec, "native_s": nat_ms[-1] / 1000 if nat_ms else None,
               "sdr": {s: sdr(ref[s], est[s]) for s in stems}}
        # The oracle's own precision floor: its TF32 run against its strict-FP32
        # run, when both are cached. A pure-FP32 port cannot sit closer to the
        # TF32 oracle than this.
        other = d / "sepcmp" / (tag[:-4] + ("tf32" if args.strict_fp32 else "fp32"))
        if all((other / f"{s}.f32").exists() for s in stems):
            row["oracle_tf32_vs_fp32"] = {
                s: sdr(np.fromfile(other / f"{s}.f32", dtype=np.float32).reshape(2, -1), ref[s]) for s in stems}
        rows.append(row)
        print(f"{d.name} [{n / SR:.1f} s, {tag}]: " + ", ".join(f"{s} {row['sdr'][s]:.1f} dB" for s in stems)
              + f"  (oracle {osec * 1000:.0f} ms, native {row['native_s'] * 1000:.0f} ms)")
        if "oracle_tf32_vs_fp32" in row:
            print("    oracle tf32 vs fp32: " + ", ".join(
                f"{s} {row['oracle_tf32_vs_fp32'][s]:.1f} dB" for s in stems)
                  + "   stem rms: " + ", ".join(f"{s} {np.sqrt((ref[s] ** 2).mean()):.1e}" for s in stems))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

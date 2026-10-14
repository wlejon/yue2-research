"""Exp 7 oracle 3: does Whisper hear the native separator's vocal stem as it
hears the oracle's?

For one run (default runs/exp7/roll_adapt_w4) over the whole song:
  1. bro-headless native/sep_stt.js separates audio.f32 with
     bro.music.loadSeparator (brosoundml's native htdemucs_ft, vocals only);
  2. the native vocal goes to 16 kHz mono exactly as sep.py makes
     stems/vocals_16k.f32 (torchaudio resample of the channel mean);
  3. Whisper large-v3 (bro.stt) transcribes every 10 s window of both;
  4. per window: the two transcripts and their word error rate, the native
     one against the oracle's (lowercased, punctuation and music notes
     dropped). Also the SDR of the native 48 kHz vocal against the oracle's
     stems/vocals.flac (sep.py's TF32 run, 24-bit FLAC).

    .venv/Scripts/python sep_whisper.py [runs/exp7/roll_adapt_w4] --bro D:/projects/bro/build/Release/bro-headless.exe
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

SR = 48000


def bro(exe: str, job: dict, path: Path):
    path.write_text(json.dumps(job), encoding="utf-8")
    env = dict(os.environ, YUE_JOB=str(path))
    res = subprocess.run([exe, "native", "native/sep_stt.js"], env=env, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0 or "done" not in res.stdout + res.stderr:
        raise SystemExit(f"bro-headless failed ({res.returncode}):\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}")
    for line in (res.stdout + res.stderr).splitlines():
        if "separated" in line or "windows" in line:
            print("  " + line.split("]")[-1].strip())


def words(t: str):
    return re.sub(r"[^a-z' ]+", " ", t.lower().replace("♪", " ")).split()


def wer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(1, len(ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="runs/exp7/roll_adapt_w4")
    ap.add_argument("--bro", default="D:/projects/bro/build/Release/bro-headless.exe")
    ap.add_argument("--json", help="write the per-window table here")
    args = ap.parse_args()
    run = Path(args.run)
    out = run / "sepcmp" / "whisper"
    out.mkdir(parents=True, exist_ok=True)

    print("separating (native, bro.music.loadSeparator)")
    bro(args.bro, {"phase": "separate", "run": str(run), "out": str(out)}, out / "job_sep.json")
    nat = np.fromfile(out / "vocals_native.f32", dtype=np.float32).reshape(2, -1)

    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    ref, sr = sf.read(run / "stems" / "vocals.flac", dtype="float32")
    ref = ref.T
    n = min(ref.shape[1], nat.shape[1])
    e = ref[:, :n].astype(np.float64) - nat[:, :n]
    sdr = 10 * np.log10((ref[:, :n].astype(np.float64) ** 2).sum() / (e ** 2).sum())
    print(f"native vocals vs sep.py's vocals.flac (TF32 oracle, 24-bit): SDR {sdr:.1f} dB over {n / SR:.1f} s")
    v16 = AF.resample(torch.from_numpy(nat).cuda().mean(0), SR, 16000).cpu().numpy().astype(np.float32)
    v16.tofile(out / "vocals_native_16k.f32")

    print("transcribing (Whisper large-v3, bro.stt)")
    bro(args.bro, {"phase": "stt", "out": str(out),
                   "inputs": {"oracle": str(run / "stems" / "vocals_16k.f32"),
                              "native": str(out / "vocals_native_16k.f32")}}, out / "job_stt.json")
    w = json.loads((out / "whisper.json").read_text(encoding="utf-8"))
    rows, same, tot_err, tot_words = [], 0, 0.0, 0
    for k, (o, nv) in enumerate(zip(w["oracle"], w["native"])):
        ro, rn = words(o), words(nv)
        r = wer(ro, rn)
        same += o == nv
        tot_err += r * max(1, len(ro))
        tot_words += max(1, len(ro))
        rows.append({"window": k, "oracle": o, "native": nv, "wer": r})
        mark = "=" if o == nv else f"WER {r:.2f}"
        print(f"[{k * 10:3d}-{k * 10 + 10:3d} s] {mark}\n    oracle: {o}" + ("" if o == nv else f"\n    native: {nv}"))
    print(f"{same}/{len(rows)} windows transcribed identically; word error rate of native against oracle "
          f"{tot_err / max(1, tot_words):.3f} over {tot_words} oracle words")
    if args.json:
        Path(args.json).write_text(json.dumps({"sdr_db": sdr, "windows": rows}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

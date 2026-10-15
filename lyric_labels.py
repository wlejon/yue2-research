"""Exp 8a: where each lyric word was sung, per frame, for the rolling-lyrics streams.

Whisper large-v3 (transformers, word timestamps) on each 10 s chunk of the
separated vocal stem; the words heard are aligned to the chunk's lyric window
(difflib over normalized words), and each matched word's span marks its
frames (25 fps) with the word's index in the window. Frames with no word
heard are -1 (not singing), frames under a heard word that matches nothing
in the window are -2 (unknown: a repeat from outside it, or a misheard word).

    .venv/Scripts/python lyric_labels.py roll_adapt_w4 roll_adapt_w6 ...
Writes runs/exp8/labels/<run>.json.
"""
import difflib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

EXP7 = Path(os.environ.get("RUNS", "runs/exp7"))     # RUNS=runs/exp8/streams for the fresh streams
OUT = Path(os.environ.get("LABELS", "runs/exp8/labels"))
FPS = 25


def words(t):
    return re.findall(r"[a-z']+", t.lower().replace("’", "'"))


def window_words(lyrics):
    """The window's sung words in order, with their line index (tags skipped)."""
    out, line = [], -1
    for l in lyrics.split("\n"):
        if not l.strip() or l.startswith("["):
            continue
        line += 1
        out += [(w, line) for w in words(l)]
    return out


def main(runs):
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
                   torch_dtype=torch.float16, device="cuda:0")
    OUT.mkdir(parents=True, exist_ok=True)
    for run in runs:
        d = EXP7 / run
        info = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        voc = np.fromfile(d / "stems" / "vocals_16k.f32", dtype=np.float32)
        chunks = []
        for c in info["chunks"]:
            f0, f1 = c["f0"], c["f1"]
            x = voc[f0 * 640: f1 * 640]
            r = asr({"raw": x, "sampling_rate": 16000}, return_timestamps="word",
                    generate_kwargs={"language": "en", "task": "transcribe"})
            heard = []
            for ch in r.get("chunks", []):
                t0, t1 = ch["timestamp"]
                if t1 is None:
                    t1 = t0 + 0.3
                for w in words(ch["text"]):
                    heard.append((w, t0, t1))
            ww = window_words(c["lyrics"])
            label = np.full(f1 - f0, -1, dtype=np.int32)
            for _, t0, t1 in heard:
                label[int(t0 * FPS): max(int(t0 * FPS) + 1, int(np.ceil(t1 * FPS)))] = -2
            sm = difflib.SequenceMatcher(a=[w for w, _, _ in heard], b=[w for w, _ in ww], autojunk=False)
            matched = 0
            for a, b, n in sm.get_matching_blocks():
                for i in range(n):
                    _, t0, t1 = heard[a + i]
                    label[int(t0 * FPS): max(int(t0 * FPS) + 1, int(np.ceil(t1 * FPS)))] = b + i
                    matched += 1
            chunks.append({"k": c["k"], "f0": f0, "f1": f1, "text": r["text"].strip(),
                           "heard": [[w, round(t0, 2), round(t1, 2)] for w, t0, t1 in heard],
                           "window": [[w, l] for w, l in ww], "matched": matched, "label": label.tolist()})
            print(f"{run} {c['k']:2d}: {len(heard):3d} heard, {matched:3d} matched of {len(ww)}  {r['text'].strip()[:70]}")
        (OUT / f"{run}.json").write_text(json.dumps({"run": run, "chunks": chunks}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])

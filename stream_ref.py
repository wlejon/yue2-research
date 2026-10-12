"""Exp 1c: windowed NAR with context, history and inpainting (reference model).

For each song, the protocol run's own plan + semantic tokens (runs/exp1/<song>_gen_W0)
are re-synthesized on the reference model over the first FRAMES frames:
  ref       whole-song NAR (one chunk over the full song, cropped), the baseline
  ref_n2    same, other noise (the natural spread)
  W<w>_L<l>_R<r>[_ip][_nc][_hist][_abs][_fut]
            windows of w frames, solving l frames of left and r frames of right
            context. Codec in the AR prefix: the solved span's (default), only
            the core's (_nc), every token from the song start to the solve's end
            (_hist, what the semantic stage's KV cache already holds), or the
            whole song (_fut, an upper bound no stream has). _ip = left context
            clamped to the latents already emitted (inpainting); _abs = latent
            position embeddings continue from the solve's first frame instead
            of restarting at 0.
Writes runs/exp1c/<song>_<variant>/ (latents + audio + seconds).

    CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python stream_ref.py <set> folk rock
"""
import sys
import time
from pathlib import Path

import torch

import meter
from lab import Lab, song_noise, save_run

FRAMES = 1500        # first 60 s
OUT = Path("runs/exp1c")

SETS = {
    # context around the window, codec of the solved span only
    "ctx": [
        dict(w=50), dict(w=50, r=25), dict(w=50, l=25), dict(w=50, l=25, ip=1), dict(w=50, l=50, ip=1),
        dict(w=50, l=25, r=12, ip=1), dict(w=50, l=50, r=25, ip=1), dict(w=25, l=50, ip=1),
        dict(w=25, l=50, r=12, ip=1), dict(w=100), dict(w=100, l=50, ip=1), dict(w=100, l=50, r=25, ip=1),
        dict(w=250, l=50, ip=1),
    ],
    # the song's codec history in the AR prefix, absolute latent positions
    "hist": [
        dict(w=50, hist=1), dict(w=50, hist=1, abs=1), dict(w=50, fut=1), dict(w=50, fut=1, abs=1),
        dict(w=50, l=25, ip=1, hist=1, abs=1), dict(w=50, l=25, r=25, ip=1, hist=1, abs=1),
        dict(w=100, l=25, r=25, ip=1, hist=1, abs=1), dict(w=250, l=25, r=25, ip=1, hist=1, abs=1),
        dict(w=250, hist=1, abs=1), dict(w=250, l=25, ip=1, hist=1),
    ],
    # how long a span does the NAR need: long windows, and short new cores
    # with a long inpainted left context
    "long": [
        dict(w=250), dict(w=500), dict(w=750), dict(w=250, l=25, r=25, ip=1), dict(w=500, l=25, r=25, ip=1),
        dict(w=50, l=250, ip=1), dict(w=50, l=250, r=25, ip=1), dict(w=100, l=400, r=25, ip=1),
        dict(w=250, l=250, r=25, ip=1), dict(w=250, l=250, r=25, ip=1, hist=1, abs=1),
    ],
    # the proposed stream: a first window for time to first audio, then long ones
    "design": [
        dict(w=[250, 500], l=25, r=25, ip=1), dict(w=[100, 250, 500], l=25, r=25, ip=1),
        dict(w=[250, 375], l=25, r=25, ip=1), dict(w=[125, 375], l=25, r=25, ip=1),
        dict(w=375, l=25, r=25, ip=1),
    ],
}


def bounds(w, T):
    sizes = w if isinstance(w, list) else [w]
    out, a, i = [], 0, 0
    while a < T:
        a += sizes[min(i, len(sizes) - 1)]
        i += 1
        if a < T:
            out.append(a)
    return out


def name(v):
    w = "-".join(map(str, v["w"])) if isinstance(v["w"], list) else v["w"]
    s = f"W{w}_L{v.get('l', 0)}_R{v.get('r', 0)}"
    for k, tag in (("ip", "_ip"), ("nc", "_nc"), ("hist", "_hist"), ("abs", "_abs"), ("fut", "_fut")):
        if v.get(k):
            s += tag
    return s


def main(set_name, songs):
    lab = Lab()
    for song in songs:
        src = meter.load_run(f"runs/exp1/{song}_gen_W0")
        plan = src.plan
        prefix = plan["prefix"]
        codec = [int(c) for c in src.semantic]
        T = len(codec)
        noise = song_noise(plan["seed"], T)
        jobs = [("ref", None), ("ref_n2", None)] + [(name(v), v) for v in SETS[set_name]]
        for vname, v in jobs:
            d = OUT / f"{song}_{vname}"
            if (d / "latents.f32").exists():
                continue
            torch.cuda.synchronize()
            t = time.perf_counter()
            if vname == "ref":
                lat = lab.nar(prefix, codec, noise)[:FRAMES]
                seams = []
            elif vname == "ref_n2":
                lat = lab.nar(prefix, codec, song_noise(2, T))[:FRAMES]
                seams = []
            else:
                lat = lab.stream(prefix, codec, noise, v["w"], v.get("l", 0), v.get("r", 0), bool(v.get("ip")),
                                 frames=FRAMES, ctx_codec=not v.get("nc"), history=bool(v.get("hist")),
                                 abs_pos=bool(v.get("abs")), future=bool(v.get("fut")))
                seams = bounds(v["w"], FRAMES)
            torch.cuda.synchronize()
            secs = time.perf_counter() - t
            audio = lab.decode(lat)
            save_run(d, latents=lat.numpy(), audio=audio,
                     info=dict(seams=seams, seconds=secs, variant=vname, frames=FRAMES))
            print(f"{song} {vname}: {secs:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])

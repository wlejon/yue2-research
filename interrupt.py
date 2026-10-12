"""Exp 3: the interrupt. Semantic tokens exist well ahead of the playhead; at
song second T0 the listener asks for "more energy". Which seam changes the
audio, by how much, and with what seam at the switch?

Base: a native exp1 run (whole-song protocol latents of its own tokens).
Switch frame P = 25*T0; the changed span is [P, P+N).

  A_lat<s>     latent offset: z[P:P+N] += s * d_latent (12-frame ramp in)
  B0           NAR re-synthesis of [P, P+N) with --left frames (250) of inpainted
               left context and --right (25) lookahead tokens, no steer (the
               re-render control: Exp 1c says a 250-frame core needs this much)
  B_nar<L>_<s> the same with the NAR energy direction at layer L, scale s
  C0           AR continuation of N tokens from codec[:P] under the same prompt
               (a re-roll control), then B0-style NAR
  C_prompt     AR continuation under the prompt with the energy pole appended
               (the NAR also sees the new prompt)
  D_ar<L>_<s>  AR continuation, same prompt, AR energy direction at layer L
               added to every continuation position

Every condition is rendered over the whole song (frames outside [P, P+N)
are the base's) and decoded; meters compare [P, P+N) and the switch.

    CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python interrupt.py folk --t0 60 --n 250
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

import meter
from lab import Lab, song_noise, save_run
from prompts import AXES

OUT = Path("runs/exp3")


def load_dir(axis, kind, layer, ev=-1):
    p = Path(f"cache/dirs/{axis}_{kind}_L{layer}_e{ev}.f32")
    return torch.from_numpy(np.fromfile(p, dtype=np.float32))


def ar_steer_hooks(lab, layer, delta, scale):
    """Forward hook adding scale*delta to decoder layer `layer`'s output (AR path)."""
    d = (delta * scale).to(lab.device, torch.bfloat16)

    def hook(mod, args, out):
        # decode steps only (one row): the prefill of prompt + song so far is untouched
        x = out[0] if isinstance(out, tuple) else out
        if x.shape[1] != 1:
            return out
        if isinstance(out, tuple):
            return (x + d,) + tuple(out[1:])
        return x + d
    return lab.model.model.layers[layer].register_forward_hook(hook)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--t0", type=float, default=60)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--axis", default="energy")
    ap.add_argument("--lat_scales", default="1,2,4")
    ap.add_argument("--nar", default="15:2,15:4")
    ap.add_argument("--ar", default="15:2,15:4")
    ap.add_argument("--left", type=int, default=250)
    ap.add_argument("--right", type=int, default=25)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--plan_only", action="store_true", help="only the C0 control and the plan-edit continuations")
    ap.add_argument("--d_only", action="store_true", help="only the base and the D (AR steer) continuations")
    ap.add_argument("--out", default=str(OUT), help="run directory (runs/exp3)")
    a = ap.parse_args()
    out_dir = Path(a.out)

    lab = Lab()
    base = meter.load_run(f"runs/exp1/{a.song}_syn_W0")
    src = meter.load_run(f"runs/exp1/{a.song}_gen_W0")
    plan = src.plan
    prefix = plan["prefix"]
    codec = [int(c) for c in src.semantic]
    T = len(codec)
    P, N = int(a.t0 * 25), a.n
    noise = song_noise(plan["seed"], T)
    z0 = torch.from_numpy(base.latents.copy())
    pos_style = plan["style"] + ", " + AXES[a.axis][0]
    abc_ids = plan["abcIds"] or None
    prefix_new = lab.prefix(pos_style, plan["lyrics"], plan["cot"], abc_ids)
    timings = {}

    def render(name, lat, codec_used=None, extra=None):
        d = out_dir / f"{a.song}_{name}"
        audio = lab.decode(lat)
        save_run(d, latents=lat.numpy(), audio=audio, semantic=codec_used,
                 info=dict(P=P, N=N, t0=a.t0, **(extra or {})))
        print(name, "done", flush=True)

    def nar_span(pref, cod, steer=None):
        torch.cuda.synchronize()
        t = time.perf_counter()
        lat = lab.window(pref, cod, noise, P, P + N, left=a.left, right=a.right, known_left=z0[:P], steer=steer)
        torch.cuda.synchronize()
        return lat, time.perf_counter() - t

    def splice(span):
        z = z0.clone()
        z[P:P + N] = span
        return z

    render("base", z0)
    if a.plan_only:
        a.lat_scales, a.nar, a.ar = "", "", ""
    if a.d_only:
        a.lat_scales, a.nar = "", ""
    # A: latent offset
    dl = torch.from_numpy(np.fromfile(f"cache/dirs/{a.axis}_latent_L0_e-1.f32", dtype=np.float32))
    ramp = torch.clamp(torch.arange(N, dtype=torch.float32) / 12, 0, 1)[:, None]
    for s in [float(x) for x in a.lat_scales.split(",") if x]:
        z = z0.clone()
        z[P:P + N] += s * ramp * dl[None]
        render(f"A_lat{s:g}", z, extra=dict(cond="A", scale=s))
    # B: NAR re-render, then NAR steer
    if not a.d_only:
        span, secs = nar_span(prefix, codec)
        timings["B0"] = secs
        render("B0", splice(span), extra=dict(cond="B0", seconds=secs))
    for spec in filter(None, a.nar.split(",")):
        L, s = spec.split(":")
        L, s = int(L), float(s)
        d = torch.stack([load_dir(a.axis, "synthesize", L, e) for e in (0, 21, 42, 63)]).mean(0)
        span, secs = nar_span(prefix, codec, steer=[dict(layer=L, delta=d, scale=s)])
        render(f"B_nar{L}_{s:g}", splice(span), extra=dict(cond="B", layer=L, scale=s, seconds=secs))
    # C: AR continuation, same prompt (re-roll) and the re-prompt
    conts = [("C0", prefix)] if a.d_only else [("C0", prefix), ("C_prompt", prefix_new)]
    if a.plan_only:
        # C_plan_*: the same continuation under an edited plan in the prefix
        # (the codec so far teacher-forced): header tempo, or section labels.
        abc = plan["abc"]
        ids0 = lab.tok.encode(abc)
        print("abc re-encode matches abcIds:", ids0 == list(plan["abcIds"]), flush=True)
        q = re.search(r"^Q:.*$", abc, re.M)
        edits = {"C_plan_tempo140": abc[:q.start()] + "Q:1/4=140" + abc[q.end():],
                 "C_plan_label": re.sub(r"^% ([a-z\-]+)", r"% \1, energetic, powerful, full band", abc, flags=re.M)}
        conts = [("C0", prefix)] + [(k, lab.prefix(plan["style"], plan["lyrics"], plan["cot"], lab.tok.encode(v)))
                                    for k, v in edits.items()]
    for name, pref in conts:
        torch.cuda.synchronize()
        t = time.perf_counter()
        cont, tm = lab.continue_semantic(pref, codec[:P], N + a.right, seed=a.seed)
        cod = codec[:P] + cont + codec[P + len(cont):]
        span, nsecs = nar_span(pref, cod)
        render(name, splice(span), codec_used=cod,
               extra=dict(cond=name, ar_seconds=tm["seconds"], prefill_seconds=tm["prefill_seconds"],
                          nar_seconds=nsecs, tokens=len(cont)))
    # D: AR steer on the continuation
    for spec in filter(None, a.ar.split(",")):
        L, s = spec.split(":")
        L, s = int(L), float(s)
        h = ar_steer_hooks(lab, L, load_dir(a.axis, "semantic", L), s)
        try:
            from yue2.sampling import generate_tokens
            from yue2.protocol import Sampling, CODEC_OFFSET
            full = list(prefix) + [c + CODEC_OFFSET for c in codec[:P]]
            ids, tm, _ = generate_tokens(lab.model, full, Sampling(min_tokens=0, max_tokens=N + a.right), a.seed, "semantic",
                                         use_cuda_graph=False)
        finally:
            h.remove()
        cont = [int(t) - CODEC_OFFSET for t in ids]
        cod = codec[:P] + cont + codec[P + len(cont):]
        span, nsecs = nar_span(prefix, cod)
        render(f"D_ar{L}_{s:g}", splice(span), codec_used=cod,
               extra=dict(cond="D", layer=L, scale=s, ar_seconds=tm["seconds"], nar_seconds=nsecs))
    (out_dir / f"{a.song}_timings.json").write_text(json.dumps(timings))


if __name__ == "__main__":
    main()

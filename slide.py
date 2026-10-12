"""Exp 4: a sliding context window for the semantic stage (reference model).

The semantic AR normally attends to prompt + plan + every codec token so far,
so its KV grows with the song and the song must end. Here the context is
bounded: the prompt + plan prefix stays as an attention sink and only the last
K codec tokens are kept. Every S tokens the context is rebuilt by one prefill
(what a native ring buffer would avoid; the attention set is the same).

  positions 'continue': the kept tokens keep their original RoPE positions
                        (a gap after the sink), new tokens continue the count
  positions 'rebase':   the context is re-laid contiguously from 0

Runs land in runs/exp4/<song>_<name>/ (plan.json + semantic.i32, in the
native harness's format, so native/gen.js can synthesize them) with
slide.json: tokens, ended, seconds, full-context NLL (mean, by third), and
repetition (share of repeated 8-grams in the last 500 tokens).

    source scripts/gpu0.sh; .venv/Scripts/python slide.py folk
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import meter
from lab import Lab
from yue2.modeling_yue2 import StaticKVCache
from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, MUSIC_END, Sampling
from yue2.sampling import distribution

OUT = Path("runs/exp4")
CONFIGS = [  # name, K (None = full context), S, positions
    ("full", None, None, "continue"),
    ("K2000", 2000, 100, "continue"),
    ("K1000", 1000, 100, "continue"),
    ("K500", 500, 100, "continue"),
    ("K250", 250, 100, "continue"),
    ("K1000_rebase", 1000, 100, "rebase"),
    ("K250_rebase", 250, 100, "rebase"),
]


@torch.inference_mode()
def sample_window(lab, prefix, K, S, positions, seed, sampling, max_tokens):
    m, dev = lab.model, lab.device
    cfg = m.config
    g = torch.Generator(device=dev).manual_seed(seed)
    hist = []                      # codec tokens with CODEC_OFFSET
    P = len(prefix)
    t0 = time.perf_counter()
    ended = False
    while len(hist) < max_tokens and not ended:
        win = hist if K is None else hist[-K:]
        ctx = list(prefix) + win
        if positions == "continue":
            pos = list(range(P)) + list(range(P + len(hist) - len(win), P + len(hist)))
        else:
            pos = list(range(len(ctx)))
        steps = max_tokens - len(hist) if S is None else S
        cache = StaticKVCache(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, len(ctx) + steps, cfg.head_dim,
                              torch.bfloat16, dev)
        out = m(torch.tensor([ctx], device=dev), position_ids=torch.tensor([pos], device=dev), past_key_values=cache,
                use_cache=True, logits_to_keep=1)
        logits = out.logits[:, -1, :]
        nxt = pos[-1] + 1
        for _ in range(steps):
            scores = distribution(logits, sampling, hist, len(hist), "semantic")
            tok = int(torch.multinomial(scores.softmax(-1), 1, generator=g).item())
            if tok == MUSIC_END:
                ended = True
                break
            hist.append(tok)
            if len(hist) >= max_tokens:
                break
            logits = m(torch.tensor([[tok]], device=dev), position_ids=torch.tensor([[nxt]], device=dev),
                       past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1, :]
            nxt += 1
    return [t - CODEC_OFFSET for t in hist], ended, time.perf_counter() - t0


@torch.inference_mode()
def full_nll(lab, prefix, codec, chunk=1024):
    """Per-token NLL of codec under the full-context model, softmax over the
    codec vocabulary + MUSIC_END (the semantic stage's allowed set)."""
    m, dev = lab.model, lab.device
    ids = list(prefix) + [c + CODEC_OFFSET for c in codec]
    hidden, _ = m.model(input_ids=torch.tensor([ids], device=dev),
                        position_ids=torch.arange(len(ids), device=dev)[None], use_cache=False)
    W = torch.cat([m.lm_head.weight[CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE], m.lm_head.weight[MUSIC_END:MUSIC_END + 1]])
    h = hidden[0, len(prefix) - 1: len(ids) - 1]
    tgt = torch.tensor(codec, device=dev)
    out = []
    for a in range(0, len(h), chunk):
        lp = F.log_softmax((h[a:a + chunk] @ W.T).float(), -1)
        out.append(-lp.gather(1, tgt[a:a + chunk, None])[:, 0])
    return torch.cat(out).cpu().numpy()


def repetition(codec, n=8, tail=500):
    t = codec[-tail:]
    grams = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    return 1 - len(set(grams)) / max(1, len(grams))


def main(songs):
    lab = Lab(load_vae=False)
    samp = Sampling(min_tokens=200, max_tokens=6000)
    for song in songs:
        src = meter.load_run(f"runs/exp1/{song}_gen_W0")
        plan = src.plan
        prefix = plan["prefix"]
        native = [int(c) for c in src.semantic]
        nll_native = full_nll(lab, prefix, native)
        rows = {"native": dict(tokens=len(native), nll=float(nll_native.mean()),
                               rep=repetition(native))}
        for name, K, S, positions in CONFIGS:
            d = OUT / f"{song}_{name}"
            if (d / "slide.json").exists():
                rows[name] = json.loads((d / "slide.json").read_text())
                continue
            codec, ended, secs = sample_window(lab, prefix, K, S, positions, 1001, samp, 6000)
            nll = full_nll(lab, prefix, codec)
            thirds = np.array_split(nll, 3)
            r = dict(K=K, S=S, positions=positions, tokens=len(codec), ended=ended, seconds=secs,
                     nll=float(nll.mean()), nll_thirds=[float(x.mean()) for x in thirds], rep=repetition(codec))
            d.mkdir(parents=True, exist_ok=True)
            np.asarray(codec, dtype=np.int32).tofile(d / "semantic.i32")
            (d / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
            (d / "slide.json").write_text(json.dumps(r))
            rows[name] = r
            print(song, name, json.dumps(r), flush=True)
        Path("tables").mkdir(exist_ok=True)
        Path(f"tables/exp4_slide_{song}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])

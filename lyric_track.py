"""Exp 8b: does YuE2 itself know which lyric word it is singing?

extract: every chunk of the rolling-lyrics streams is re-run teacher-forced on
the reference model with the context it was sampled from (prompt with the
chunk's lyric window, cot off, then its memory and its own codec tokens).
For each of the chunk's frames it keeps every head's attention to each word
of the window (the softmax mass over the word's tokens) and the residual
after every second layer.

eval: against the Whisper word labels (lyric_labels.py), per frame:
  * each head alone: the word it attends to most (exact / within one word /
    the right line), at its best lag;
  * a probe over all heads' attention to a word (448 numbers per word, one
    linear score, softmax over the window's words), trained leave-one-run-out;
  * a linear probe on the residual: singing or not, and the line of the
    window (0..5), leave-one-run-out.

    .venv/Scripts/python lyric_track.py extract roll_adapt_w4 ...
    .venv/Scripts/python lyric_track.py eval roll_adapt_w4 ...
Writes runs/exp8/track/<run>/<k>.npz and tables/exp8_lyric_track.md.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

EXP7 = Path(os.environ.get("RUNS", "runs/exp7"))     # RUNS + JOB: runs/exp8/streams + jobs/exp8.json
JOB = Path(os.environ.get("JOB", "jobs/exp7.json"))
LABELS = Path("runs/exp8/labels")
TRACK = Path("runs/exp8/track")
RES_LAYERS = list(range(1, 28, 2))      # residual after layers 1, 3, ..., 27
LAGS = range(-12, 13, 2)


def lyric_word_spans(tok, ids_text, text, lyrics_at):
    """Token column ranges of each sung word of the lyrics part of `text`
    (encoded as ids_text), in order, with their line index."""
    ends, n = [], 0
    for t in ids_text:
        n += len(tok.decode([t]))
        ends.append(n)
    ends = np.array(ends)
    spans, line, pos = [], -1, lyrics_at
    for l in text[lyrics_at:].split("\n"):
        if l.strip() and not l.startswith("["):
            line += 1
            for m in re.finditer(r"[a-z']+", l.lower().replace("’", "'")):
                a, b = pos + m.start(), pos + m.end()
                t0 = int(np.searchsorted(ends, a, side="right"))
                t1 = int(np.searchsorted(ends, b, side="left")) + 1
                spans.append((m.group(), line, t0, t1))
        pos += len(l) + 1
    return spans


@torch.inference_mode()
def extract(runs):
    from lab import Lab
    from yue2.protocol import CODEC_OFFSET, EOD, SongRequest
    from yue2.nar import attention
    lab = Lab(load_vae=False)
    bb = lab.model.model
    dev = lab.device
    for run in runs:
        d = EXP7 / run
        info = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        job = json.loads(JOB.read_text(encoding="utf-8"))
        tokens = np.fromfile(d / "tokens.i32", dtype=np.int32)
        out = TRACK / run
        out.mkdir(parents=True, exist_ok=True)
        for c in info["chunks"]:
            style = job["styles"][c["style"]]["style"]
            req = SongRequest(style=style, lyrics=c["lyrics"], cot="off")
            text = req.text()
            ids_text = lab.tok.encode(text)
            prefix = lab.prefix(style, c["lyrics"], "off")
            assert prefix[0] == EOD and prefix[1:1 + len(ids_text)] == ids_text
            spans = lyric_word_spans(lab.tok, ids_text, text, text.index("[Lyrics]\n") + len("[Lyrics]\n"))
            f0, f1 = c["f0"], c["f1"]
            s0 = max(0, f0 - c["memory"])
            ids = prefix + [int(t) + CODEC_OFFSET for t in tokens[s0:f1]]
            T, P = len(ids), len(prefix)
            rows = torch.arange(P + f0 - s0, T, device=dev)
            W = len(spans)
            M = torch.zeros(T, W, device=dev)
            for w, (_, _, t0, t1) in enumerate(spans):
                M[1 + t0: 1 + t1, w] = 1          # +1: the EOD in front of the text
            pos = torch.arange(T, device=dev)[None]
            cos, sin = bb.rotary_emb(pos)
            x = bb.embed_tokens(torch.tensor([ids], device=dev))
            att, res = [], []
            for li, layer in enumerate(bb.layers):
                q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
                qr = q[0, rows].float().transpose(0, 1)                       # H, R, D
                kk = k[0].float().repeat_interleave(q.shape[2] // k.shape[2], 1).transpose(0, 1)   # H, T, D
                s = qr @ kk.transpose(1, 2) / qr.shape[-1] ** 0.5
                s = s.masked_fill(pos[0][None, None, :] > rows[None, :, None], float("-inf"))
                att.append((s.softmax(-1) @ M).half().cpu())                    # H, R, W
                h = attention(q[0], k[0], v[0], causal=True)
                x = x + layer.self_attn.o_proj(h.flatten(1)[None])
                x = x + layer.mlp(layer.post_attention_layernorm(x))
                if li in RES_LAYERS:
                    res.append(x[0, rows].half().cpu())
            np.savez(out / f"{c['k']}.npz", att=torch.stack(att).numpy(), res=torch.stack(res).numpy(),
                     line=np.array([l for _, l, _, _ in spans]), words=np.array([w for w, _, _, _ in spans]))
            print(f"{run} {c['k']:2d}: {T} tokens, {W} words", flush=True)


def load(runs):
    data = []
    for run in runs:
        lab = json.loads((LABELS / f"{run}.json").read_text(encoding="utf-8"))
        for c in lab["chunks"]:
            z = np.load(TRACK / run / f"{c['k']}.npz")
            words = [w for w, _ in c["window"]]
            assert list(z["words"]) == words, (run, c["k"])
            data.append({"run": run, "k": c["k"], "att": z["att"].astype(np.float32),
                         "res": z["res"].astype(np.float32), "line": z["line"],
                         "label": np.array(c["label"])})
    return data


def shift(label, lag):
    """label at frame t - lag: a head that runs `lag` frames behind the singing."""
    out = np.full_like(label, -3)
    if lag >= 0:
        out[lag:] = label[:len(label) - lag]
    else:
        out[:lag] = label[-lag:]
    return out


def head_scores(data):
    L, H = data[0]["att"].shape[:2]
    best = {}
    for lag in LAGS:
        hit = np.zeros((L, H)); near = np.zeros((L, H)); ln = np.zeros((L, H)); n = 0
        for c in data:
            lab = shift(c["label"], lag)
            m = lab >= 0
            if not m.any():
                continue
            pred = c["att"][:, :, m].argmax(-1)                 # L, H, n
            y = lab[m]
            hit += (pred == y).sum(-1)
            near += (np.abs(pred - y) <= 1).sum(-1)
            ln += (c["line"][pred] == c["line"][y]).sum(-1)
            n += m.sum()
        for li in range(L):
            for h in range(H):
                if hit[li, h] / n > best.get((li, h), (0,))[0]:
                    best[(li, h)] = (hit[li, h] / n, near[li, h] / n, ln[li, h] / n, lag)
    return sorted(best.items(), key=lambda kv: -kv[1][0]), n


def train_softmax(X, Y, groups, classes=None, wd=1e-3, steps=400):
    """Linear model. classes=None: X[i] is (W_i, F) per-candidate features and
    Y[i] the right candidate (a pointer); else X is (N, F), Y class ids."""
    dev = "cuda"
    if classes is None:
        F = X[0].shape[1]
        w = torch.zeros(F, device=dev, requires_grad=True)
        b = None
    else:
        F = X.shape[1]
        w = torch.zeros(F, classes, device=dev, requires_grad=True)
        b = torch.zeros(classes, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w] + ([b] if b is not None else []), max_iter=steps, line_search_fn="strong_wolfe")
    if classes is None:
        Wm = max(len(x) for x in X)
        Xp = torch.zeros(len(X), Wm, F, device=dev)
        pad = torch.ones(len(X), Wm, dtype=torch.bool, device=dev)
        for i, x in enumerate(X):
            Xp[i, :len(x)] = torch.tensor(x, device=dev)
            pad[i, :len(x)] = False
        Ys = torch.tensor(Y, device=dev)

        def loss_fn():
            s = (Xp @ w).masked_fill(pad, float("-inf"))
            return torch.nn.functional.cross_entropy(s, Ys) + wd * (w * w).sum()
    else:
        Xt, Yt = torch.tensor(X, device=dev), torch.tensor(Y, device=dev)

        def loss_fn():
            return torch.nn.functional.cross_entropy(Xt @ w + b, Yt) + wd * (w * w).sum()

    def closure():
        opt.zero_grad()
        lo = loss_fn()
        lo.backward()
        return lo
    opt.step(closure)
    return w.detach(), (b.detach() if b is not None else None)


def pointer_probe(data, lag):
    """All heads' attention to a word as its features; leave-one-run-out."""
    runs = sorted({c["run"] for c in data})
    hit = near = ln = n = 0
    for held in runs:
        Xtr, Ytr = [], []
        for c in data:
            if c["run"] == held:
                continue
            lab = shift(c["label"], lag)
            for t in np.nonzero(lab >= 0)[0][::2]:
                Xtr.append(np.log(1e-4 + c["att"][:, :, t].reshape(-1, c["att"].shape[-1]).T))
                Ytr.append(lab[t])
        w, _ = train_softmax(Xtr, Ytr, None)
        for c in data:
            if c["run"] != held:
                continue
            lab = shift(c["label"], lag)
            m = lab >= 0
            if not m.any():
                continue
            f = torch.tensor(np.log(1e-4 + c["att"][:, :, m].reshape(-1, m.sum(), c["att"].shape[-1])), device="cuda")
            pred = torch.einsum("fnw,f->nw", f, w).argmax(-1).cpu().numpy()
            y = lab[m]
            hit += (pred == y).sum(); near += (np.abs(pred - y) <= 1).sum()
            ln += (c["line"][pred] == c["line"][y]).sum(); n += m.sum()
    return hit / n, near / n, ln / n, n


def residual_probes(data, layer_i):
    """Singing (a word under the frame) vs not, and the window line sung (0..5)."""
    runs = sorted({c["run"] for c in data})
    sing_ok = sing_n = line_ok = line_n = 0
    for held in runs:
        Xs, Ys, Xl, Yl = [], [], [], []
        for c in data:
            if c["run"] == held:
                continue
            lab = c["label"]
            r = c["res"][layer_i]
            k = lab != -2
            Xs.append(r[k]); Ys.append((lab[k] >= 0).astype(np.int64))
            m = lab >= 0
            Xl.append(r[m]); Yl.append(np.minimum(c["line"][lab[m]], 5))
        X, Y = np.concatenate(Xs), np.concatenate(Ys)
        mu, sd = X.mean(0), X.std(0) + 1e-3
        ws, bs = train_softmax((X - mu) / sd, Y, None, classes=2)
        X2, Y2 = np.concatenate(Xl), np.concatenate(Yl)
        wl, bl = train_softmax((X2 - mu) / sd, Y2, None, classes=6)
        for c in data:
            if c["run"] != held:
                continue
            lab = c["label"]
            r = torch.tensor((c["res"][layer_i] - mu) / sd, device="cuda")
            k = lab != -2
            ps = (r @ ws + bs).argmax(-1).cpu().numpy()
            sing_ok += (ps[k] == (lab[k] >= 0)).sum(); sing_n += k.sum()
            m = lab >= 0
            pl = (r @ wl + bl).argmax(-1).cpu().numpy()
            line_ok += (pl[m] == np.minimum(c["line"][lab[m]], 5)).sum(); line_n += m.sum()
    return sing_ok / sing_n, line_ok / line_n


def evaluate(runs):
    rolled = [r for r in runs if r != "sheet_long"]
    out = ["# Exp 8: does YuE2 know which lyric word it is singing?", ""]
    for name, rs in (("rolling windows (4-6 lines)", rolled), ("the whole 40-line sheet", ["sheet_long"])):
        rs = [r for r in rs if r in runs]
        if not rs:
            continue
        data = load(rs)
        ranked, n = head_scores(data)
        out += [f"## Single heads, {name}: {n} sung frames", "",
                "| layer.head | right word | within one word | right line | lag (frames) |", "|---|---|---|---|---|"]
        for (li, h), (a, b, c, lag) in ranked[:12]:
            out.append(f"| {li}.{h} | {a:.0%} | {b:.0%} | {c:.0%} | {lag} |")
        out.append("")
        print("\n".join(out[-16:]), flush=True)
    data = load(rolled)
    ranked, _ = head_scores(data)
    lag = ranked[0][1][3]
    a, b, c, n = pointer_probe(data, lag)
    out += [f"## A probe over every head's attention (leave one run out, rolling windows, lag {lag})", "",
            f"{n} sung frames: right word {a:.0%}, within one word {b:.0%}, right line {c:.0%}", ""]
    print("\n".join(out[-4:]), flush=True)
    out += ["## Linear probes on the residual (leave one run out, rolling windows)", "",
            "| after layer | singing or not | line of the window |", "|---|---|---|"]
    for i, li in enumerate(RES_LAYERS):
        s, l = residual_probes(data, i)
        out.append(f"| {li} | {s:.0%} | {l:.0%} |")
        print(out[-1], flush=True)
    Path("tables").mkdir(exist_ok=True)
    Path("tables/exp8_lyric_track.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def roll(runs, head=(14, 10)):
    """What the rolling window needs: per chunk, the furthest window line sung.
    Truth: the stem's Whisper labels, a line sung when 60% of its words were
    heard. The head: the line of the word it attends to most, a line sung when
    it pointed into 60% of the line's words (2+ frames each) while its mass on
    the lyrics was at least `gate`. Against it, what the stream uses now:
    Whisper on the mix (chunks.json heardLines, inside the window)."""
    li, h = head
    data = load([r for r in runs if r != "sheet_long"])
    rows = []
    for c in data:
        info = json.loads((EXP7 / c["run"] / "chunks.json").read_text(encoding="utf-8"))["chunks"][c["k"]]
        a, b = info["lyricLines"]
        nl = int(c["line"].max()) + 1
        lab = c["label"]
        wl = [np.nonzero(c["line"] == l)[0] for l in range(nl)]
        truth = max([l for l in range(nl) if np.isin(wl[l], lab[lab >= 0]).mean() >= 0.6], default=-1)
        mix = max([i - a for i in info.get("heardLines", []) if a <= i < b], default=-1)
        att = c["att"][li, h]                                  # R, W
        rows.append((truth, mix, att.argmax(-1), att.sum(-1), wl, c["line"]))
    out = ["", f"## The rolling window's question per chunk: the furthest window line sung ({len(rows)} chunks)", "",
           "| reader | right | one line off | missed a sung line | heard a line not sung |", "|---|---|---|---|---|"]

    def score(name, preds):
        t = np.array([r[0] for r in rows]); p = np.array(preds)
        out.append(f"| {name} | {np.mean(p == t):.0%} | {np.mean(np.abs(p - t) == 1):.0%} | "
                   f"{np.mean(p < t):.0%} | {np.mean(p > t):.0%} |")
    score("Whisper on the mix (the stream now)", [r[1] for r in rows])
    for gate in (0.05, 0.1, 0.2, 0.3):
        preds = []
        for _, _, ptr, mass, wl, line in rows:
            on = mass >= gate
            cnt = np.bincount(ptr[on], minlength=len(line))
            preds.append(max([l for l in range(len(wl)) if (cnt[wl[l]] >= 2).mean() >= 0.6], default=-1))
        score(f"head {li}.{h}, lyric mass >= {gate}", preds)
    print("\n".join(out))
    with open("tables/exp8_lyric_track.md", "a", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def check(runs, head=(14, 10), lag=-2):
    """The head picked on exp7, unchanged, on other streams: frames, then roll()."""
    li, h = head
    data = load(runs)
    out = ["", f"## Head {li}.{h} on fresh streams ({EXP7}: {', '.join(runs)}), lag {lag}", "",
           "| stream | sung frames | right word | within one word | right line |", "|---|---|---|---|---|"]
    tot = np.zeros(4)
    for run in runs + ["all"]:
        if run == "all":
            n, a, b, c = tot
        else:
            n = a = b = c = 0
            for x in data:
                if x["run"] != run:
                    continue
                lab = shift(x["label"], lag)
                m = lab >= 0
                pred = x["att"][li, h][m].argmax(-1)
                y = lab[m]
                n += m.sum(); a += (pred == y).sum(); b += (np.abs(pred - y) <= 1).sum()
                c += (x["line"][pred] == x["line"][y]).sum()
            tot += (n, a, b, c)
        out.append(f"| {run} | {int(n)} | {a / max(n, 1):.0%} | {b / max(n, 1):.0%} | {c / max(n, 1):.0%} |")
    print("\n".join(out))
    with open("tables/exp8_lyric_track.md", "a", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    roll(runs, head)


if __name__ == "__main__":
    {"extract": extract, "eval": evaluate, "roll": roll, "check": check}[sys.argv[1]](sys.argv[2:])

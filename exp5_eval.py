"""Exp 5 analysis: endless streams of 10 s chunks (native/endless.js).

Every stream dir holds chunks.json, latents.f32, audio.f32 and (after
native/clap_embed.js) clap.f32, one CLAP embedding per 10 s chunk.

  - cost: per-chunk splice (semantic), inpaint (NAR) and decode seconds, and
    the GPU duty they make against 10 s of audio;
  - breaks: chunks whose mean latent norm leaves the corpus range 6.5..9.5,
    and chunks under -30 LUFS (the music stopped);
  - seams: spectral flux at chunk joins in z units of the stream's own flux
    (the max over +-2 columns, against that statistic everywhere else), and the
    share of joins above its 95th percentile (exp1's audible-step criterion);
  - beat carry: the beat grid of the 6 s before a join extended over the 4 s
    after it, and the onset strength landing on it (beat_carry), at joins and
    at mid-chunk points of the same streams;
  - coherence (constant styles): CLAP cosine between consecutive chunks, CLAP
    cosine to the style's reference centroid (first and last third), the
    chunks whose nearest reference is another style, tempo folded to the
    asked bpm's octave (percent off);
  - mobility (the tour): each chunk's nearest style among the four reference
    centroids; after a change X -> Y, the share of Y's chunks heard as Y, and
    the first chunk that is.

    .venv/Scripts/python exp5_eval.py runs/exp5 --out tables/exp5_endless.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import meter
from prompts import STREAM_STYLES

FPS = 25
COLS = 4            # log-mel columns per latent frame
REF_STYLES = ["folk", "synth", "metal", "lofi"]


def load(d: Path):
    info = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
    lat = np.fromfile(d / "latents.f32", dtype=np.float32).reshape(-1, 64)
    a = np.fromfile(d / "audio.f32", dtype=np.float32)
    audio = a.reshape(2, -1)
    clap = None
    if (d / "clap.f32").exists():
        clap = np.fromfile(d / "clap.f32", dtype=np.float32).reshape(-1, 512)
    return dict(name=d.name, info=info, lat=lat, audio=audio, clap=clap)


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def seam_stats(s):
    C = s["info"]["chunk"]
    T = s["info"]["frames"]
    lm = meter.logmel(s["audio"])
    flux = meter.spectral_flux(lm)
    n = len(flux)
    # the max over a 5-column neighbourhood, everywhere
    pad = np.pad(flux, 2, mode="edge")
    loc = np.max(np.stack([pad[i:i + n] for i in range(5)]), axis=0)
    joins = np.arange(C, T, C) * COLS
    joins = joins[joins < n - 2]
    mask = np.ones(n, bool)
    for j in joins:
        mask[max(0, j - 50):j + 50] = False
    mask[:100] = False
    ref = loc[mask]
    z = (loc[joins] - ref.mean()) / (ref.std() + 1e-9)
    p95 = np.percentile(ref, 95)
    return dict(seam_z=float(z.mean()), seam_z_max=float(z.max()) if len(z) else float("nan"),
                above95=float((loc[joins] > p95).mean()) if len(z) else float("nan"))


ENV_FPS = 22050 / 512


def beat_carry(env, t):
    """Does the beat carry on across time t? The beat period and phase from
    the 6 s before t (autocorrelation peak in 0.3..1.2 s, then the phase
    whose comb collects the most onset strength), the grid extended over the
    4 s after t, and the onset strength on it (max within one frame of each
    predicted beat) over the mean strength there: about 1 at random phase,
    above it when onsets keep landing on the predicted grid."""
    a, b, c = int((t - 6) * ENV_FPS), int(t * ENV_FPS), int((t + 4) * ENV_FPS)
    if a < 0 or c > len(env):
        return float("nan")
    w = env[a:b] - env[a:b].mean()
    ac = np.correlate(w, w, mode="full")[len(w) - 1:]
    lags = np.arange(int(0.3 * ENV_FPS), int(1.2 * ENV_FPS))
    P = lags[np.argmax(ac[lags])]
    phases = [env[a + ph:b:P].sum() for ph in range(P)]
    ph = int(np.argmax(phases))
    first = a + ph + int(np.ceil((b - (a + ph)) / P)) * P
    hits = [env[max(0, i - 1):i + 2].max() for i in range(first, c, P)]
    return float(np.mean(hits) / (env[b:c].mean() + 1e-9))


def beat_continuity(s, skip=()):
    import librosa
    y = librosa.resample(meter.mono(s["audio"]), orig_sr=meter.SR, target_sr=22050)
    env = librosa.onset.onset_strength(y=y, sr=22050, hop_length=512)
    C = s["info"]["chunk"] / FPS
    K = len(s["info"]["chunks"])
    joins = [beat_carry(env, k * C) for k in range(1, K) if k not in skip]
    mids = [beat_carry(env, k * C + C / 2) for k in range(1, K) if k not in skip]
    return dict(beat_join=float(np.nanmedian(joins)), beat_mid=float(np.nanmedian(mids)))


def chunk_features(s):
    C = s["info"]["chunk"] / FPS
    return [meter.features(meter.segment(s["audio"], k * C, (k + 1) * C)) for k in range(len(s["info"]["chunks"]))]


def bpm_of(style):
    m = re.search(r"(\d+)\s*bpm", style)
    return int(m.group(1)) if m else None


def folded_off(tempo, asked):
    """Percent between a tempo and the asked bpm, octaves folded (half and double time count as on tempo)."""
    o = np.log2(tempo / asked)
    return float(100 * (2 ** abs(o - np.round(o)) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="tables/exp5_endless.md")
    args = ap.parse_args()
    root = Path(args.root)
    streams = {d.name: load(d) for d in sorted(root.iterdir()) if (d / "chunks.json").exists()}

    # reference centroids per style, from the ref streams' chunks 1.. (chunk 0 starts from nothing)
    refs = {}
    for key in REF_STYLES:
        es = [streams[n]["clap"][1:] for n in streams if n.startswith(f"ref_{key}_") and streams[n]["clap"] is not None]
        refs[key] = dict(mu=unit(np.concatenate(es).mean(0)), chunks=np.concatenate(es))
    mus = np.stack([refs[k]["mu"] for k in REF_STYLES])
    nearest = lambda e: [REF_STYLES[i] for i in np.argmax(e @ mus.T, axis=1)]
    ref_self = {k: float((v["chunks"] @ v["mu"]).mean()) for k, v in refs.items()}
    ref_acc = {k: float(np.mean([x == k for x in nearest(v["chunks"])])) for k, v in refs.items()}

    rows, mob = {}, {}
    for n, s in streams.items():
        info, ch = s["info"], s["info"]["chunks"]
        keys = [c["style"] for c in ch]
        kind = n.split("_")[0]
        r = dict(kind=kind, memory=info["memory"], cfg=info.get("cfgScale"), bridge=info.get("bridge", False),
                 chunks=len(ch))
        for f in ("spliceSeconds", "inpaintSeconds", "decodeSeconds"):
            r[f] = float(np.mean([c[f] for c in ch]))
        r["duty"] = (r["spliceSeconds"] + r["inpaintSeconds"] + r["decodeSeconds"]) / (info["chunk"] / FPS)
        if r["bridge"]:
            b = [c for c in ch if c.get("bridge")]
            f = [c for c in ch if c.get("fresh")]
            r["bridge_s"] = float(np.mean([c["spliceSeconds"] for c in b]))
            r["fresh_s"] = float(np.mean([c["spliceSeconds"] for c in f]))
        norms = np.array([c["latentNorm"] for c in ch])
        r["norm_min"], r["norm_max"] = float(norms[1:].min()), float(norms[1:].max())
        r["breaks"] = int(((norms[1:] < 6.5) | (norms[1:] > 9.5)).sum())
        feats = chunk_features(s)
        lufs = np.array([f["lufs"] for f in feats])
        r["silent"] = int((~(lufs > -30)).sum())
        r["first_silent"] = int(np.argmax(~(lufs > -30))) if r["silent"] else -1
        r.update(seam_stats(s))
        r.update(beat_continuity(s))
        e = s["clap"]
        r["clap_consec"] = float(np.mean((e[1:] * e[:-1]).sum(1)))
        lab = nearest(e)
        r["labels"] = lab
        if len(set(keys)) == 1 and keys[0] in refs:
            cs = e @ refs[keys[0]]["mu"]
            third = max(1, len(cs) // 3)
            r["clap_ref_first"] = float(cs[1:1 + third].mean())
            r["clap_ref_last"] = float(cs[-third:].mean())
            r["off_style"] = int(sum(x != keys[0] for x in lab[1:]))
            asked = bpm_of(STREAM_STYLES[keys[0]]["style"])
            offs = [folded_off(f["tempo"], asked) for f in feats[1:] if f["tempo"] > 0]
            r["tempo_off_med"] = float(np.median(offs))
            r["tempo_on"] = float(np.mean(np.array(offs) < 4))
        if kind == "tour":
            key = ("bridge" if r["bridge"] else "plain") + (f" cfg {r['cfg']:g}" if r["cfg"] else "") + f" m{r['memory']}"
            for k in range(1, len(keys)):
                if keys[k] == keys[k - 1]:
                    continue
                Y = keys[k]
                if lab[k - 1] == Y:      # already heard as Y (the tour's return to a folk it never left)
                    continue
                js =[j for j in range(4) if k + j < len(keys) and keys[k + j] == Y]
                hit = [lab[k + j] == Y for j in js]
                mob.setdefault(key, []).append(dict(stream=n, change=f"{keys[k - 1]}->{Y}", hits=hit,
                                                    first=next((j for j, h in zip(js, hit) if h), None)))
        rows[n] = r

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(dict(rows=rows, ref_self=ref_self, ref_acc=ref_acc, mobility=mob),
                                                   indent=1), encoding="utf-8")

    def avg(rs, k):
        v = [r[k] for r in rs if k in r and r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        return float(np.mean(v)) if v else float("nan")

    L = ["# Exp 5: endless streams of 10 s chunks", "",
         f"Streams in `{root}`; `exp5_eval.py`. Memory = tokens of history each chunk is opened by (25 per second). "
         "Every chunk: splice (semantic) + inpaint (NAR) + decode on one 4090, INT8 AR, 32 ODE steps.", ""]
    L += ["## Cost per 10 s chunk (plain streams, mean)", "",
          "| memory | splice s | inpaint s | decode s | GPU duty |", "|---|---|---|---|---|"]
    plain = [r for r in rows.values() if not r["cfg"] and not r["bridge"]]
    for m in sorted({r["memory"] for r in plain}):
        rs = [r for r in plain if r["memory"] == m]
        L.append(f"| {m} | {avg(rs, 'spliceSeconds'):.2f} | {avg(rs, 'inpaintSeconds'):.2f} | "
                 f"{avg(rs, 'decodeSeconds'):.2f} | {avg(rs, 'duty'):.0%} |")
    br = [r for r in rows.values() if r["bridge"]]
    cf = [r for r in rows.values() if r["cfg"] and not r["bridge"]]
    if br:
        L.append(f"\nA bridge: the fresh clip {avg(br, 'fresh_s'):.2f} s + the transition (4 candidates, scored) "
                 f"{avg(br, 'bridge_s'):.2f} s of semantic sampling.")
    if cf:
        L.append(f"With CFG (two branches): splice {avg(cf, 'spliceSeconds'):.2f} s per chunk.")
    L += ["", "## Seams and the beat", "",
          "Seam z: flux at a join in units of the stream's own flux spread (0 = like anywhere else). "
          "Beat carry: onset strength on the beat grid extended across a point, over the mean (about 1 = random phase).", "",
          "| group | streams | seam z | seam z max | joins > p95 | beat carry at joins | beat carry mid-chunk |",
          "|---|---|---|---|---|---|---|"]
    groups = {}
    for n, r in rows.items():
        g = (r["kind"] + (" bridge" if r["bridge"] else "") + (f" cfg {r['cfg']:g}" if r["cfg"] else "")
             + (f" m{r['memory']}" if r["kind"] in ("const", "tour") else ""))
        groups.setdefault(g, []).append(r)
    for g in sorted(groups):
        rs = groups[g]
        L.append(f"| {g} | {len(rs)} | {avg(rs, 'seam_z'):.2f} | {max(r['seam_z_max'] for r in rs):.2f} | "
                 f"{avg(rs, 'above95'):.2f} | {avg(rs, 'beat_join'):.2f} | {avg(rs, 'beat_mid'):.2f} |")
    L += ["", "## Constant styles: does a stream stay itself?", "",
          f"Reference streams (fresh starts, 2 seeds per style): self-similarity "
          + ", ".join(f"{k} {v:.3f}" for k, v in ref_self.items()) + "; their chunks' nearest centroid is their own style "
          + ", ".join(f"{k} {v:.0%}" for k, v in ref_acc.items()) + ".", "",
          "| stream | chunks | norm min..max | breaks | silent chunks (first) | CLAP consecutive | CLAP to ref first / last | chunks heard as another style | tempo off (median %, share within 4%) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, r in rows.items():
        if "clap_ref_first" not in r:
            continue
        L.append(f"| {n} | {r['chunks']} | {r['norm_min']:.2f}..{r['norm_max']:.2f} | {r['breaks']} | "
                 f"{r['silent']} ({r['first_silent']}) | {r['clap_consec']:.3f} | {r['clap_ref_first']:.3f} / {r['clap_ref_last']:.3f} | "
                 f"{r['off_style']} | {r['tempo_off_med']:.1f}, {r['tempo_on']:.0%} |")
    L += ["", "## Mobility: after a style change, which style is heard", "",
          "The tour: folk, synth, metal, lofi, folk, 4 chunks (40 s) each. Each change X -> Y: the share of Y's 4 chunks whose "
          "nearest reference is Y, and the chunk the first such arrives (0 = the first chunk after the change). Up to 8 changes per row "
          "(2 seeds); a change whose last chunk before it was already heard as Y is left out.", "",
          "| condition | Y heard (share of chunks) | changes reached in 40 s | first arrival (median) | per change (hits) |",
          "|---|---|---|---|---|"]
    for key in sorted(mob):
        ms = mob[key]
        share = np.mean([np.mean(m["hits"]) for m in ms])
        reached = np.mean([m["first"] is not None for m in ms])
        firsts = [m["first"] for m in ms if m["first"] is not None]
        per = " ".join(m["change"].split("->")[1][:2] + ":" + "".join("x" if h else "." for h in m["hits"]) for m in ms)
        L.append(f"| {key} | {share:.0%} | {reached:.0%} | {np.median(firsts) if firsts else float('nan'):.1f} | {per} |")
    L += ["", "## Every stream's nearest style per chunk", "", "```"]
    for n, r in rows.items():
        if r["kind"] == "tour":
            L.append(f"{n:<32} " + " ".join(x[:2] for x in r["labels"]))
    L.append(f"{'(asked)':<32} " + " ".join(k[:2] for k, c in [("folk", 4), ("synth", 4), ("metal", 4), ("lofi", 4), ("folk", 4)] for _ in range(c)))
    L += ["```", ""]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

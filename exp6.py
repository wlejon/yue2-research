"""Exp 6 analysis: one score, many styles. Can the vocal of one render ride
over the backing of another?

Every render in runs/exp6 is <song>_<style>_s<seed>, the same ABC (exp1's
folk or rock score) in another style or seed, separated by sep.py
(stems/vocals.flac; backing = mix - vocals).

  - tempo: beat-tracked tempo of each render against the score's Q:;
  - alignment: DTW of the vocal stems' chroma (CENS) between two renders of
    one score. drift(t) = where render j is when render i is at t. Reported:
    median and p90 |drift| over sung frames, the linear fit's slope (a global
    tempo ratio), and p90 of what is left after that fit (|residual|: how much a
    single global stretch leaves unaligned);
  - harmony: chroma cosine of render i's vocal against render j's backing
    over i's sung frames, j warped onto i by the DTW path: matched (j = i),
    cross (same score, other style or seed), cross unwarped, and other-score
    (a null: the other song's backing, unwarped);
  - demos: folk_orig_s1's vocal over each other folk render's backing, raw and
    after one global time stretch (runs/exp6/demos, for listening).

    .venv/Scripts/python exp6.py runs/exp6 --out tables/exp6_shared_score.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

SR = 48000
SR2 = 22050
HOP = 1024
FT = HOP / SR2      # seconds per feature frame


def load(d: Path):
    import librosa
    import soundfile as sf
    mix = np.fromfile(d / "audio.f32", dtype=np.float32).reshape(2, -1)
    voc, _ = sf.read(d / "stems" / "vocals.flac", dtype="float32")
    voc = voc.T[:, :mix.shape[1]]
    back = mix[:, :voc.shape[1]] - voc
    m = lambda x: librosa.resample(x.mean(0), orig_sr=SR, target_sr=SR2)
    yv, yb, ym = m(voc), m(back), m(mix)
    v_cens = librosa.feature.chroma_cens(y=yv, sr=SR2, hop_length=HOP)
    v_chroma = librosa.feature.chroma_cqt(y=yv, sr=SR2, hop_length=HOP)
    b_chroma = librosa.feature.chroma_cqt(y=yb, sr=SR2, hop_length=HOP)
    rms = librosa.feature.rms(y=yv, frame_length=2048, hop_length=HOP)[0]
    n = min(v_cens.shape[1], v_chroma.shape[1], b_chroma.shape[1], len(rms))
    rdb = 20 * np.log10(rms[:n] + 1e-9)
    sung = rdb > rdb.max() - 30
    _, beats = librosa.beat.beat_track(y=ym, sr=SR2, hop_length=512, units="time")
    tempo = float(60 / np.median(np.diff(beats))) if len(beats) > 8 else float("nan")
    pj = json.loads((d / "plan.json").read_text(encoding="utf-8"))
    q = re.search(r"Q:\s*1/4\s*=\s*(\d+)", pj.get("abc", ""))
    return dict(name=d.name, dir=d, voc=voc, back=back, v_cens=v_cens[:, :n], v_chroma=v_chroma[:, :n],
                b_chroma=b_chroma[:, :n], sung=sung, tempo=tempo, q=int(q.group(1)) if q else None)


def dtw_map(a, b):
    """For every frame of a, the matching frame of b (DTW on vocal CENS)."""
    import librosa
    _, wp = librosa.sequence.dtw(X=a["v_cens"], Y=b["v_cens"], metric="cosine",
                                 global_constraints=True, band_rad=0.1)
    wp = wp[::-1]
    n = a["v_cens"].shape[1]
    m = np.zeros(n)
    for i in range(n):
        js = wp[wp[:, 0] == i, 1]
        m[i] = np.median(js) if len(js) else np.nan
    idx = np.arange(n)
    ok = ~np.isnan(m)
    return np.interp(idx, idx[ok], m[ok])


def vocal_env(r):
    """The vocal stem's onset envelope at 100 per second (z-scored)."""
    import librosa
    y = librosa.resample(r["voc"].mean(0), orig_sr=SR, target_sr=SR2)
    e = librosa.onset.onset_strength(y=y, sr=SR2, hop_length=SR2 // 100)
    return (e - e.mean()) / (e.std() + 1e-9)


def local_lags(ea, eb, win=10.0, hop=5.0, search=3.0):
    """Per 10 s window of a (5 s hop): the lag of b (seconds, + = b later)
    maximizing the normalized cross-correlation within +-3 s, and that peak."""
    W, H, S = int(win * 100), int(hop * 100), int(search * 100)
    out = []
    for a0 in range(S, len(ea) - W - S, H):
        x = ea[a0:a0 + W]
        if x.std() < 0.3:                # a window with no singing
            continue
        best, lag = -1.0, 0
        for d in range(-S, S + 1, 1):
            yb = eb[a0 + d:a0 + d + W]
            if len(yb) < W:
                continue
            c = float(np.dot(x - x.mean(), yb - yb.mean()) / (W * x.std() * yb.std() + 1e-9))
            if c > best:
                best, lag = c, d
        out.append((a0 / 100, lag / 100, best))
    return np.array(out)


def unit_cols(c):
    return c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)


def harmony(a, b, jmap):
    j = np.clip(np.round(jmap).astype(int), 0, b["b_chroma"].shape[1] - 1)
    va, bb = unit_cols(a["v_chroma"]), unit_cols(b["b_chroma"])
    cs = (va * bb[:, j]).sum(0)
    return float(cs[a["sung"]].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="tables/exp6_shared_score.md")
    ap.add_argument("--no-demos", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    rs = {d.name: load(d) for d in sorted(root.iterdir()) if (d / "stems" / "vocals.flac").exists()}
    songs = sorted({n.split("_")[0] for n in rs})

    envs = {n: vocal_env(r) for n, r in rs.items()}
    rows = []
    for song in songs:
        names = sorted(n for n in rs if n.startswith(song + "_"))
        for i, j in itertools.permutations(names, 2):
            a, b = rs[i], rs[j]
            jm = dtw_map(a, b)
            t = np.arange(len(jm)) * FT
            drift = jm * FT - t
            s = a["sung"]
            fit = np.polyfit(t[s], (jm * FT)[s], 1)
            resid = (jm * FT)[s] - np.polyval(fit, t[s])
            lin = np.clip(np.round(np.arange(len(jm)) * b["v_cens"].shape[1] / len(jm)).astype(int), 0, None)
            same_style = i.rsplit("_", 1)[0] == j.rsplit("_", 1)[0]
            ll = local_lags(envs[i], envs[j])
            good = ll[ll[:, 2] > 0.5] if len(ll) else ll
            lag = dict(lag_windows=int(len(good)), lag_of=int(len(ll)))
            if len(good):
                lg = good[:, 1]
                lag.update(lag_med=float(np.median(np.abs(lg))), lag_p90=float(np.percentile(np.abs(lg), 90)),
                           within50=float(np.mean(np.abs(lg) <= 0.05)),
                           lag_spread=float(np.percentile(lg, 90) - np.percentile(lg, 10)),
                           lag_profile=[[float(t), float(d)] for t, d, _ in good])
            rows.append(dict(song=song, i=i, j=j, same_style=same_style, **lag,
                             drift_med=float(np.median(np.abs(drift[s]))), drift_p90=float(np.percentile(np.abs(drift[s]), 90)),
                             slope=float(fit[0]), icpt=float(fit[1]), resid_p90=float(np.percentile(np.abs(resid), 90)),
                             harm_cross=harmony(a, b, jm), harm_unwarped=harmony(a, b, lin)))
    matched = {n: harmony(r, r, np.arange(r["v_cens"].shape[1])) for n, r in rs.items()}
    null = []
    for i in rs:
        for j in rs:
            if i.split("_")[0] != j.split("_")[0]:
                b = rs[j]
                lin = np.clip(np.round(np.arange(rs[i]["v_cens"].shape[1]) * b["v_cens"].shape[1] / rs[i]["v_cens"].shape[1]).astype(int), 0, None)
                null.append(harmony(rs[i], b, lin))

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(dict(rows=rows, matched=matched, null=null,
        tempo={n: dict(tempo=r["tempo"], q=r["q"]) for n, r in rs.items()}), indent=1), encoding="utf-8")

    L = ["# Exp 6: one score, many styles", "",
         f"Renders in `{root}` (`exp6.py`): the same ABC per song, cot full with the score supplied; vocals separated by htdemucs_ft.", "",
         "## Tempo (beat-tracked) against the score's Q:", "", "| render | tempo | Q: |", "|---|---|---|"]
    for n, r in rs.items():
        L.append(f"| {n} | {r['tempo']:.1f} | {r['q']} |")
    L += ["", "## Alignment between two renders of one score (vocal-stem DTW, seconds)", "",
          "Pairs grouped: same style other seed, and other style. |drift| over sung frames; slope = the global tempo ratio of a linear fit; "
          "resid p90 = what one global stretch leaves.", "",
          "| song | pairs | drift median | drift p90 | slope range | resid p90 median (max) |", "|---|---|---|---|---|---|"]
    for song in songs:
        for same in (True, False):
            g = [r for r in rows if r["song"] == song and r["same_style"] == same]
            if not g:
                continue
            L.append(f"| {song} {'same style, other seed' if same else 'other style'} | {len(g)} | "
                     f"{np.median([r['drift_med'] for r in g]):.3f} | {np.median([r['drift_p90'] for r in g]):.3f} | "
                     f"{min(r['slope'] for r in g):.4f}..{max(r['slope'] for r in g):.4f} | "
                     f"{np.median([r['resid_p90'] for r in g]):.3f} ({max(r['resid_p90'] for r in g):.3f}) |")
    L += ["", "## Alignment by windowed cross-correlation of the vocal stems' onset envelopes", "",
          "Per 10 s window (5 s hop) of render i, the lag of render j within +-3 s that best matches its vocal onsets; "
          "windows with a correlation peak over 0.5 only (no DTW, so repeated choruses cannot be confused). "
          "within 50 ms = share of windows where j is within 50 ms of i; spread = p90 - p10 of the lag "
          "(a constant offset has spread 0 and one shift fixes it).", "",
          "| song | pairs | windows kept | median abs lag | p90 abs lag | within 50 ms | lag spread median |", "|---|---|---|---|---|---|---|"]
    for song in songs:
        for same in (True, False):
            g = [r for r in rows if r["song"] == song and r["same_style"] == same and r.get("lag_windows")]
            if not g:
                continue
            L.append(f"| {song} {'same style, other seed' if same else 'other style'} | {len(g)} | "
                     f"{np.mean([r['lag_windows'] / r['lag_of'] for r in g]):.0%} | {np.median([r['lag_med'] for r in g]):.3f} | "
                     f"{np.median([r['lag_p90'] for r in g]):.3f} | {np.mean([r['within50'] for r in g]):.0%} | "
                     f"{np.median([r['lag_spread'] for r in g]):.3f} |")
    L += ["", "## Harmony: render i's vocal against render j's backing (chroma cosine over sung frames)", "",
          "| condition | mean | min..max |", "|---|---|---|",
          f"| matched (its own backing) | {np.mean(list(matched.values())):.3f} | {min(matched.values()):.3f}..{max(matched.values()):.3f} |"]
    for lab, key in (("cross, DTW-warped", "harm_cross"), ("cross, unwarped", "harm_unwarped")):
        v = [r[key] for r in rows]
        L.append(f"| {lab} (same score) | {np.mean(v):.3f} | {min(v):.3f}..{max(v):.3f} |")
    L.append(f"| other song's backing (null) | {np.mean(null):.3f} | {min(null):.3f}..{max(null):.3f} |")
    L += ["", "## Pairs", "", "| i | j | drift med | drift p90 | slope | resid p90 | harmony warped / unwarped |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['i']} | {r['j']} | {r['drift_med']:.3f} | {r['drift_p90']:.3f} | {r['slope']:.4f} | {r['resid_p90']:.3f} | "
                 f"{r['harm_cross']:.3f} / {r['harm_unwarped']:.3f} |")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))

    if not args.no_demos:
        # The score fixes the tempo, so two renders differ by an offset: b
        # shifted by the median windowed lag lines its vocal onsets up with a's.
        import soundfile as sf
        dd = root / "demos"
        dd.mkdir(exist_ok=True)
        for base in ("folk_orig_s1", "rock_orig_s1"):
            if base not in rs:
                continue
            a = rs[base]
            song = base.split("_")[0]
            for j in rs:
                if not j.startswith(song + "_") or j == base:
                    continue
                b = rs[j]
                r = next(x for x in rows if x["i"] == base and x["j"] == j)
                if not r.get("lag_profile"):
                    continue
                off = int(round(np.median([x[1] for x in r["lag_profile"]]) * SR))
                bb = b["back"][:, off:] if off >= 0 else np.pad(b["back"], ((0, 0), (-off, 0)))
                m = min(a["voc"].shape[1], bb.shape[1])
                sf.write(dd / f"{base}_vocal_over_{j}_backing.flac", (a["voc"][:, :m] + bb[:, :m]).T, SR,
                         subtype="PCM_24")
        print("demos in", dd)


if __name__ == "__main__":
    main()

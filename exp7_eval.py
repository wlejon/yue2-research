"""Exp 7 analysis: style changes that keep the beat, scored tours against
scored references, rolling lyrics.

  - per style change X -> Y on a tour: the share of Y's chunks whose nearest
    reference centroid (CLAP) is Y, and beat carry (exp5_eval.beat_carry) at
    the change's joins: the join into the change chunk and, for a bridge, the
    join into the fresh clip; the worst of them is the change's. Unscored tours
    are labelled against exp5's ref_* streams, scored tours (stour_*;
    exp5c's score_tour_* are the same streams as stour_*_s1, bit for bit, so
    left out) against exp7's scoref_* (the same score, one style each);
  - lyrics: Whisper on the separated vocal of each 10 s window (stt.json).
    A sheet line is sung in a window when 60% of its words are in the
    transcript; per stream the distinct lines sung, the furthest line, lines
    sung again in a later window (not the next), the transcript words found
    nowhere in the sheet, and for rolled streams the sung lines that were in
    that chunk's window.

    .venv/Scripts/python exp7_eval.py --out tables/exp7_stream.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import meter
from exp5_eval import ENV_FPS, FPS, beat_carry, load, unit
from prompts import LYRICS

PLAIN = ["folk", "synth", "metal", "lofi"]
SCORED = ["synth", "metal", "jazz", "orch"]


def envelope(s):
    import librosa
    y = librosa.resample(meter.mono(s["audio"]), orig_sr=meter.SR, target_sr=22050)
    return librosa.onset.onset_strength(y=y, sr=22050, hop_length=512)


def centroids(streams, prefix, keys):
    refs = {}
    for k in keys:
        es = [s["clap"][1:] for n, s in streams.items() if n.startswith(f"{prefix}{k}_s") and s["clap"] is not None]
        if es:
            refs[k] = unit(np.concatenate(es).mean(0))
    return refs


def condition(n):
    n = re.sub(r"_s\d+$", "", n)
    return n.replace("score_tour", "stour")


def changes(s, refs, env):
    info, ch = s["info"], s["info"]["chunks"]
    C = info["chunk"] / FPS
    keys = [c["style"].replace("score_", "") for c in ch]
    names = list(refs)
    mus = np.stack([refs[k] for k in names])
    lab = [names[i] for i in np.argmax(s["clap"] @ mus.T, axis=1)]
    out, change_joins = [], set()
    for k in range(1, len(keys)):
        if keys[k] == keys[k - 1]:
            continue
        joins = [k] + ([k + 1] if ch[k].get("bridge") else [])
        change_joins.update(joins)
        if lab[k - 1] == keys[k]:
            continue
        Y = keys[k]
        js = [j for j in range(4) if k + j < len(keys) and keys[k + j] == Y]
        bc = [beat_carry(env, j * C) for j in joins]
        out.append(dict(change=f"{keys[k - 1]}->{Y}", hits=[lab[k + j] == Y for j in js],
                        beat=float(np.nanmin(bc)), beat_joins=bc))
    other = [beat_carry(env, j * C) for j in range(1, len(keys)) if j not in change_joins]
    return out, float(np.nanmedian(other)) if other else float("nan"), lab


SCORE_BEAT = 60 / 90     # the folk score's Q:1/4=90; every scored render holds it exactly


def comb_phase(env, period, t0, t1):
    """Absolute beat phase (beats, 1/64 steps) of the onset envelope between
    t0 and t1 seconds: the comb at `period` collecting the most strength,
    interpolated at fractional envelope frames."""
    best, arg = -1.0, 0.0
    for q in range(64):
        x = np.arange(q / 64 * period, t1, period)
        x = x[x >= t0]
        v = np.interp(x * ENV_FPS, np.arange(len(env)), env).mean()
        if v > best:
            best, arg = v, q / 64
    return arg


def phase_offsets(s, env, period=SCORE_BEAT):
    """Under a score every chunk has the score's tempo, so a join either keeps
    the beat's phase or jumps it. The comb phase at the score's period of the
    6 s before and of the 6 s after each join, and the circular distance
    between them in beats: 0 = the beat goes on, 0.5 = on the off-beat
    (uniform in 0..0.5 at random). Returns {join chunk: offset}. (A period
    rounded to whole envelope frames, 0.673 s, leaves a 0.08-beat bias.)"""
    C = s["info"]["chunk"] / FPS
    T = len(env) / ENV_FPS
    out = {}
    for j in range(1, len(s["info"]["chunks"])):
        t = j * C
        if t - 6 < 0 or t + 6 > T:
            continue
        d = abs(comb_phase(env, period, t - 6, t) - comb_phase(env, period, t, t + 6)) % 1
        out[j] = min(d, 1 - d)
    return out, period


WORD = re.compile(r"[a-z']+")


def words(t):
    return WORD.findall(t.lower().replace("’", "'"))


def sung_lines(text):
    return [l for l in text.splitlines() if l.strip() and not l.startswith("[")]


def lyric_stats(s, sheet, rolled):
    stt = json.loads((Path(s["dir"]) / "stt.json").read_text(encoding="utf-8"))
    wins = stt.get("vocals") or stt["mix"]
    lines = sheet
    lw = [words(l) for l in lines]
    vocab = set(w for l in lw for w in l)
    per, alien, total = [], 0, 0
    for t in wins:
        tw = words(t)
        bag = set(tw)
        per.append([i for i, l in enumerate(lw) if l and sum(w in bag for w in l) / len(l) >= 0.6])
        alien += sum(w not in vocab for w in tw)
        total += len(tw)
    seen, again = {}, 0
    for w, ls in enumerate(per):
        for i in ls:
            if i in seen and w - seen[i] > 1:
                again += 1
            seen[i] = w
    r = dict(windows=len(wins), words=total, distinct=len(seen), furthest=max(seen) if seen else -1,
             again=again, alien=alien / max(1, total), lines=per)
    if rolled:
        on = tot = 0
        for w, ls in enumerate(per):
            if w >= len(s["info"]["chunks"]):
                break
            a, b = s["info"]["chunks"][w]["lyricLines"]
            on += sum(a <= i < b for i in ls)
            tot += len(ls)
        r["in_window"] = on / max(1, tot)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="runs/exp5,runs/exp7")
    ap.add_argument("--out", default="tables/exp7_stream.md")
    args = ap.parse_args()
    streams = {}
    for r in args.roots.split(","):
        for d in sorted(Path(r).iterdir()):
            if (d / "chunks.json").exists() and (d / "clap.f32").exists():
                s = load(d)
                s["dir"] = str(d)
                streams[d.name] = s
    plain_refs = centroids(streams, "ref_", PLAIN)
    score_refs = centroids(streams, "scoref_", SCORED)

    # the scored references: does each style stay itself under the score?
    ref_acc = {}
    names = list(score_refs)
    mus = np.stack([score_refs[k] for k in names])
    for k in names:
        es = np.concatenate([s["clap"][1:] for n, s in streams.items() if n.startswith(f"scoref_{k}_s")])
        ref_acc[k] = float(np.mean(np.argmax(es @ mus.T, axis=1) == names.index(k)))

    tours = {n: s for n, s in streams.items()
             if re.match(r"tour_m250(_bridge|_drop\d+)?_s\d$", n) or n.startswith("stour_")}
    res = {}
    for n, s in sorted(tours.items()):
        refs = score_refs if s["info"]["chunks"][0]["style"].startswith("score_") else plain_refs
        env = envelope(s)
        ch, base, lab = changes(s, refs, env)
        res[n] = dict(changes=ch, beat_other=base, labels=lab)
        if refs is score_refs:
            offs, period = phase_offsets(s, env)
            info = s["info"]["chunks"]
            cj = {k for k in range(1, len(info)) if info[k]["style"] != info[k - 1]["style"]}
            fj = {k + 1 for k in cj if info[k].get("bridge")}
            res[n]["phase"] = dict(period=period, change=[offs[j] for j in sorted(cj) if j in offs],
                                   fresh=[offs[j] for j in sorted(fj) if j in offs],
                                   trims=[info[j].get("trim") for j in sorted(fj)],
                                   other=[v for j, v in offs.items() if j not in cj | fj])

    lyr = {}
    long_sheet = sung_lines(LYRICS["long"])
    river = []
    for l in sung_lines(LYRICS["river"]):
        if l not in river:
            river.append(l)
    for n, s in streams.items():
        if not (Path(s["dir"]) / "stt.json").exists():
            continue
        if n.startswith("roll_") or n == "sheet_long":
            lyr[n] = lyric_stats(s, long_sheet, n.startswith("roll_"))
        elif n in ("voice_m500", "voice_m1000"):
            lyr[n] = lyric_stats(s, river, False)

    out = Path(args.out)
    out.with_suffix(".json").write_text(json.dumps(dict(tours=res, lyrics=lyr, score_ref_acc=ref_acc), indent=1),
                                        encoding="utf-8")
    L = ["# Exp 7: style changes that keep the beat; rolling lyrics", "",
         "`exp7_eval.py`. Tours: memory 250 (10 s), 10 s chunks. Beat carry = onset strength on the beat grid of the 6 s "
         "before a join extended over the 4 s after it, over the mean (about 1 = random phase); a change's is the worst of its joins "
         "(the change chunk's, and a bridge's fresh clip's).", "",
         "Scored references (folk score, one style each, 2 seeds): chunks heard as their own style "
         + ", ".join(f"{k} {v:.0%}" for k, v in ref_acc.items()) + ".", "",
         "| condition | streams | changes | Y heard (share of chunks) | reached in 40 s | beat carry at the change (median, min) | elsewhere |",
         "|---|---|---|---|---|---|---|"]
    conds = {}
    for n, r in res.items():
        conds.setdefault(condition(n), []).append(r)
    for c in sorted(conds):
        rs = conds[c]
        ch = [x for r in rs for x in r["changes"]]
        if not ch:
            continue
        share = np.mean([np.mean(x["hits"]) for x in ch])
        reached = np.mean([any(x["hits"]) for x in ch])
        bt = [x["beat"] for x in ch if not np.isnan(x["beat"])]
        L.append(f"| {c} | {len(rs)} | {len(ch)} | {share:.0%} | {reached:.0%} | {np.median(bt):.2f}, {np.min(bt):.2f} | "
                 f"{np.nanmedian([r['beat_other'] for r in rs]):.2f} |")
    L += ["", "Scored tours, beat phase across joins (beats, 0 = the beat goes on, 0.5 = the off-beat; "
          "random is uniform 0..0.5, median 0.25):", "",
          "| condition | beat period s | into the change chunk (median; each) | into the fresh clip (median; each; trims) | other joins (median; share < 0.1) |",
          "|---|---|---|---|---|"]
    for c in sorted(conds):
        ps = [r["phase"] for n, r in res.items() if condition(n) == c and "phase" in r]
        if not ps:
            continue
        chg = [x for p in ps for x in p["change"]]
        fr = [x for p in ps for x in p["fresh"]]
        tr = [x for p in ps for x in p["trims"]]
        oth = [x for p in ps for x in p["other"]]
        fcell = (f"{np.median(fr):.2f}; " + " ".join(f"{x:.2f}" for x in fr)
                 + ("; trims " + " ".join(str(x) for x in tr) if any(x is not None for x in tr) else "")) if fr else "-"
        L.append(f"| {c} | {np.median([p['period'] for p in ps]):.3f} | {np.median(chg):.2f}; "
                 + " ".join(f"{x:.2f}" for x in chg) + f" | {fcell} | {np.median(oth):.2f}; {np.mean(np.array(oth) < 0.1):.0%} |")
    L += ["", "Per chunk, nearest style:", "", "```"]
    for n, r in sorted(res.items()):
        L.append(f"{n:<28} " + " ".join(x[:2] for x in r["labels"]))
    L += ["```", "", "## Lyrics", "",
          f"Whisper large-v3 on the separated vocal per 10 s window. voice_m*: exp5's river sheet ({len(river)} distinct lines), "
          f"the others the long sheet ({len(long_sheet)} lines, none repeated); roll_rN: N lines further each chunk, 4 shown.", "",
          "| stream | windows | words | distinct lines sung | furthest line | sung again later | words not in the sheet | sung lines in the chunk's window |",
          "|---|---|---|---|---|---|---|---|"]
    for n, r in sorted(lyr.items()):
        L.append(f"| {n} | {r['windows']} | {r['words']} | {r['distinct']} | {r['furthest']} | {r['again']} | "
                 f"{r['alien']:.0%} | {r.get('in_window', float('nan')):.0%} |")
    L += ["", "Lines sung per window (sheet line indices):", "", "```"]
    for n, r in sorted(lyr.items()):
        L.append(f"{n:<14} " + " ".join("[" + ",".join(map(str, ls)) + "]" for ls in r["lines"]))
    L += ["```", ""]
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

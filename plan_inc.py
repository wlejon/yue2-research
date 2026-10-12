"""Exp 4b: can the plan (the ABC score) be written incrementally, section by
section, ahead of the semantic stream? (reference model, AR 'abc' phase)

The score is sectioned by '% <name>' comment lines (one per [Section] of the
lyrics). For each song and seed:
  full      the plan of the full lyrics (the protocol)
  partial   the plan of the lyrics cut after section CUT only
  extended  partial's score cut back to its first CUT sections, then continued
            under the full lyrics prompt (the radio appending sections later)
Compared on: the section list vs the lyrics' sections, the header (M/L/Q/K),
bars per section, chord vocabulary overlap with full, and truncation.

    source scripts/gpu0.sh; .venv/Scripts/python plan_inc.py folk synthpop
"""
import json
import re
import sys
from pathlib import Path

import torch

from lab import Lab
from prompts import SONGS
from yue2.protocol import ABC_START, EOD, GenerationConfig, SongRequest
from yue2.sampling import generate_tokens

CUT = 4


def sections(abc):
    out, cur = [], None
    for line in abc.splitlines():
        if line.startswith("% "):
            cur = dict(name=line[2:].strip(), bars=0, chords=set())
            out.append(cur)
        elif cur is not None and line.startswith('"'):
            # vocal lines carry the chords; count their bars only (the Ins voice repeats the same bars)
            cur["bars"] += line.count("|")
            cur["chords"].update(re.findall(r'"([^"]+)"', line))
    return out


def header(abc):
    return {k: (re.search(rf"^{k}:(.*)$", abc, re.M) or [None, None])[1] for k in "MLQK"}


def lyric_sections(lyrics):
    return [m.lower() for m in re.findall(r"^\[([^\]]+)\]", lyrics, re.M)]


def cut_lyrics(lyrics, n):
    idx = [m.start() for m in re.finditer(r"^\[", lyrics, re.M)]
    return lyrics[: idx[n]] if n < len(idx) else lyrics


def cut_abc(abc, n):
    idx = [m.start() for m in re.finditer(r"^% ", abc, re.M)]
    return abc[: idx[n]] if n < len(idx) else abc


def plan(lab, style, lyrics, seed, abc_prefix=""):
    req = SongRequest(style=style, lyrics=lyrics, cot="full", seed=seed)
    base = [EOD] + lab.tok.encode(req.text()) + [ABC_START]
    pre_ids = lab.tok.encode(abc_prefix) if abc_prefix else []
    ids, timing, trunc = generate_tokens(lab.model, base + pre_ids, GenerationConfig().abc, seed, "abc")
    return abc_prefix + lab.tok.decode(ids), timing, trunc


def summary(abc, lyrics, full=None):
    secs = sections(abc)
    chords = set().union(*[s["chords"] for s in secs]) if secs else set()
    r = dict(sections=[s["name"] for s in secs], bars=[s["bars"] for s in secs], header=header(abc),
             lyric_sections=lyric_sections(lyrics))
    r["sections_match"] = [s.split()[0] for s in r["sections"]] == [s.split()[0] for s in r["lyric_sections"]]
    if full is not None:
        fc = set().union(*[s["chords"] for s in sections(full)])
        r["chord_jaccard_vs_full"] = len(chords & fc) / max(1, len(chords | fc))
        r["header_same_as_full"] = header(abc) == header(full)
    return r


def main(songs):
    lab = Lab(load_vae=False)
    out = {}
    for song in songs:
        s = SONGS[song]
        for seed in (1, 2):
            full, t_full, tr_full = plan(lab, s["style"], s["lyrics"], seed)
            part_lyr = cut_lyrics(s["lyrics"], CUT)
            part, t_part, tr_part = plan(lab, s["style"], part_lyr, seed)
            ext, t_ext, tr_ext = plan(lab, s["style"], s["lyrics"], seed + 100, cut_abc(part, CUT))
            out[f"{song}_s{seed}"] = dict(
                full=dict(summary(full, s["lyrics"]), truncated=tr_full, seconds=t_full["seconds"]),
                partial=dict(summary(part, part_lyr, full), truncated=tr_part, seconds=t_part["seconds"]),
                extended=dict(summary(ext, s["lyrics"], full), truncated=tr_ext, seconds=t_ext["seconds"]))
            d = Path("runs/exp4b") / f"{song}_s{seed}"
            d.mkdir(parents=True, exist_ok=True)
            for k, v in (("full", full), ("partial", part), ("extended", ext)):
                (d / f"{k}.abc").write_text(v, encoding="utf-8")
            print(json.dumps(out[f"{song}_s{seed}"], default=list), flush=True)
    Path("tables/exp4b_plan.json").write_text(json.dumps(out, indent=1, default=list))


if __name__ == "__main__":
    main(sys.argv[1:])

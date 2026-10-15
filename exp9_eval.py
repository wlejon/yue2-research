"""Exp 9: a window rolled by the model's own lyric pointer (head 14.10, read
natively while each chunk is sampled; brosoundml Yue2Endless), end to end.

Measured as FINDINGS 10 measures lyrics: Whisper large-v3 on the separated
vocal of each 10 s window (sep.py, native/stt.js: stt.json); a sheet line is
sung in a window when 60% of its words are in the transcript. Per stream:
distinct lines sung, the furthest line, lines sung again in a later window
(not the next), transcript words found nowhere in the sheet, and the sung
lines that were inside that chunk's window (exp7_eval.lyric_stats).

Against: FINDINGS 10's rows on the same setup ('voice' style, the 40-line
sheet, memory 500, 24 chunks: runs/exp7 sheet_long, roll_r1_w4, roll_r2_w4,
roll_adapt_w4 (Whisper on the mix), roll_adapt_w6) and the six exp8 streams
(runs/exp8/streams, the same voices, sheets and seeds, rolled one line a
chunk). Also, per head-rolled stream, how far the head's own reading ran
(lines reached) against where the stem's Whisper heard the stream.

    .venv/Scripts/python exp9_eval.py [--out tables/exp9_lyrics.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from exp7_eval import lyric_stats, sung_lines
from prompts import LYRICS

EXP9 = Path("runs/exp9/streams")
EXP8 = Path("runs/exp8/streams")
EXP7 = Path("runs/exp7")
VOICES = ["rock_m_harbor", "folk_m_desert", "soul_f_harbor", "country_m_desert", "edm_f_desert", "ballad_f_harbor"]


def stream(d):
    return {"dir": str(d), "info": json.loads((d / "chunks.json").read_text(encoding="utf-8"))}


def stats(d, sheet, rolled):
    if not (d / "stt.json").exists():
        return None
    return lyric_stats(stream(d), sheet, rolled)


def head_reach(d):
    """Per chunk, the head's furthest line reached (or -1) and how many lines it moved the window."""
    ch = json.loads((d / "chunks.json").read_text(encoding="utf-8"))["chunks"]
    return [max(c.get("linesReached") or [-1]) for c in ch]


def row(name, r, note=""):
    if r is None:
        return f"| {name} | (no stt.json) |||||||"
    return (f"| {name} | {r['windows']} | {r['words']} | {r['distinct']} | {r['furthest']} | {r['again']} | "
            f"{r['alien']:.0%} | {r.get('in_window', float('nan')):.0%} | {note} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tables/exp9_lyrics.md")
    args = ap.parse_args()
    job8 = json.loads(Path("jobs/exp8.json").read_text(encoding="utf-8"))
    long_sheet = sung_lines(LYRICS["long"])
    hdr = ["| stream | windows | words | distinct lines sung | furthest line | sung again later | words not in the sheet | "
           "sung lines in the chunk's window | |", "|---|---|---|---|---|---|---|---|---|"]
    L = ["# Exp 9: lyrics rolled by the model's own pointer (head 14.10), end to end", "",
         "Whisper large-v3 on the separated vocal (htdemucs_ft) per 10 s window, a line sung at 60% of its words.", "",
         f"## The FINDINGS 10 setup: 'voice', the 40-line sheet ({len(long_sheet)} lines), memory 500, 24 chunks", ""] + hdr
    for n, note in (("sheet_long", "the whole sheet"), ("roll_r1_w4", "4 lines, 1 further each chunk"),
                    ("roll_r2_w4", "4 lines, 2 further each chunk"),
                    ("roll_adapt_w4", "4 lines after the last line Whisper heard on the mix (+ 3-chunk stall rule)"),
                    ("roll_adapt_w6", "6 lines, the same")):
        L.append(row(n, stats(EXP7 / n, long_sheet, n.startswith("roll_")), note))
    L.append(row("**head_w4**", stats(EXP9 / "head_w4", long_sheet, True),
                 "**4 lines after the furthest line head 14.10 reached (no Whisper, no stall rule)**"))
    reach = {}
    if (EXP9 / "head_w4" / "chunks.json").exists():
        reach["head_w4"] = head_reach(EXP9 / "head_w4")
    L += ["", "## The six exp8 voices (14 chunks, their sheets, the same seeds)", ""] + hdr
    tot = {"old": [], "new": []}
    for n in VOICES:
        st = next(s for s in job8["streams"] if s["name"] == n)
        sheet = [l for l in job8["sheets"][st["lyricsRoll"]["sheet"]] if l.strip() and not l.startswith("[")]
        a, b = stats(EXP8 / n, sheet, True), stats(EXP9 / n, sheet, True)
        L.append(row(n, a, "1 line further each chunk (exp8)"))
        L.append(row(f"**{n}**", b, "**head-rolled**"))
        if a and b:
            tot["old"].append(a)
            tot["new"].append(b)
        if (EXP9 / n / "chunks.json").exists():
            reach[n] = head_reach(EXP9 / n)
    for k, rs in tot.items():
        if rs:
            L.append(f"| {'mean, ' + ('1 line a chunk' if k == 'old' else 'head-rolled')} | | | "
                     f"{sum(r['distinct'] for r in rs) / len(rs):.1f} | {sum(r['furthest'] for r in rs) / len(rs):.1f} | "
                     f"{sum(r['again'] for r in rs) / len(rs):.1f} | {sum(r['alien'] for r in rs) / len(rs):.0%} | "
                     f"{sum(r['in_window'] for r in rs) / len(rs):.0%} | |")
    L += ["", "## Where the head moved each window (furthest line reached per chunk, -1 = none)", "", "```"]
    for n, r in reach.items():
        L.append(f"{n:<18} " + " ".join(map(str, r)))
    L += ["```", ""]
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

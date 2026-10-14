"""Write native job files (jobs/*.json) for the experiments.

    .venv/Scripts/python make_jobs.py exp1
"""
import json
import sys
from pathlib import Path

from prompts import SONGS

ROOT = Path(__file__).parent
JOBS = ROOT / "jobs"


def write(name, job):
    JOBS.mkdir(exist_ok=True)
    (JOBS / f"{name}.json").write_text(json.dumps(job, indent=1, ensure_ascii=False), encoding="utf-8")
    n = len(job.get("runs", job.get("streams", [])))
    print("wrote", JOBS / f"{name}.json", n, "runs" if "runs" in job else "streams")


def exp1():
    """Streaming: protocol vs windowed generate (timing), and windowed synthesis
    of the protocol run's own tokens (seams)."""
    gen, syn = [], []
    for song, s in SONGS.items():
        base = dict(style=s["style"], lyrics=s["lyrics"], cot="full", seed=1001)
        gen.append(dict(name=f"{song}_gen_W0", **base, narChunkFrames=0))
        gen.append(dict(name=f"{song}_gen_W50", **base, narChunkFrames=50))
        for w in (0, 25, 50, 100, 250):
            syn.append(dict(name=f"{song}_syn_W{w}", mode="synthesize", **{"from": f"runs/exp1/{song}_gen_W0"},
                            narChunkFrames=w))
        # the natural spread: the same tokens, whole-song NAR, other noise
        syn.append(dict(name=f"{song}_syn_W0n2", mode="synthesize", **{"from": f"runs/exp1/{song}_gen_W0"},
                        narChunkFrames=0, seed=2))
    write("exp1_gen", dict(out="runs/exp1", arWeights="int8", vae=True, runs=gen))
    write("exp1_syn", dict(out="runs/exp1", arWeights="int8", vae=True, runs=syn))


def exp1d():
    """Two processes on one GPU: semantic sampling alone, NAR (500-frame
    windows) alone, then both at once (suffix _conc)."""
    s = SONGS["folk"]
    for tag in ("", "_conc"):
        write(f"exp1d_sem{tag}", dict(out="runs/exp1d", arWeights="int8", vae=False, runs=[
            dict(name=f"sem{tag}", style=s["style"], lyrics=s["lyrics"], cot="full", seed=1001, runSynthesize=False)]))
        write(f"exp1d_nar{tag}", dict(out="runs/exp1d", arWeights="int8", vae=False, runs=[
            dict(name=f"nar{tag}", mode="synthesize", **{"from": "runs/exp1/folk_gen_W0"}, narChunkFrames=500)]))


def vram():
    """One process per scenario (native/vram.js), so the caching allocator of
    one cannot hide the next."""
    s = SONGS["folk"]
    song = dict(style=s["style"], lyrics=s["lyrics"])
    scen = {
        "base": dict(),
        "vae": dict(vae=True),
        "bf16": dict(yue="bf16"),
        "int8": dict(yue="int8"),
        "int8_vae": dict(yue="int8", vae=True),
        "gen_default": dict(yue="int8", vae=True, generate=dict()),
        "gen_bf16": dict(yue="bf16", vae=True, generate=dict()),
        "gen_max4500": dict(yue="int8", vae=True, generate=dict(semanticSampling=dict(maxTokens=4500))),
        "gen_cfg": dict(yue="int8", vae=True, generate=dict(cfgScale=1.5)),
        "gen_W500": dict(yue="int8", vae=True, generate=dict(narChunkFrames=500)),
        "gen_novae": dict(yue="int8", generate=dict()),
    }
    for name, sc in scen.items():
        write(f"vram_{name}", dict(scenario=name, out=f"tables/vram/{name}.json", song=song, runs=[], **sc))


def exp4c():
    """Mid-plan edits followed by the semantic stage (native, abc supplied, the
    same seed): a tempo change, a key change, and a section label naming the
    energy, each from the folk plan's '% bridge' on."""
    s = SONGS["folk"]
    abc = json.loads((ROOT / "runs/exp1/folk_gen_W0/plan.json").read_text(encoding="utf-8"))["abc"]
    i = abc.index("% bridge")
    tail = abc[i:]
    edits = {
        "tempo140": abc[:i] + "Q:1/4=140\n" + tail,
        "keyEm": abc[:i] + "K:Em\n" + tail,
        "label": abc[:i] + tail.replace("% chorus", "% chorus, energetic, powerful, full band"),
        "orig": abc,
    }
    runs = [dict(name=f"folk_{k}", style=s["style"], lyrics=s["lyrics"], cot="full", seed=1001, abc=v)
            for k, v in edits.items()]
    write("exp4c", dict(out="runs/exp4c", arWeights="int8", vae=True, runs=runs))


CAPTURE_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
NAR_EVALS = [0, 21, 42, 63]


def exp2():
    """Labelled corpus: genre x axis x pole x seed, with AR (semantic) and NAR
    residual means captured."""
    from prompts import GENRES, AXES, LYRICS
    runs = []
    for seed in (11, 12):
        for g, base in GENRES.items():
            runs.append(dict(name=f"{g}_neutral_s{seed}", style=base, lyrics=LYRICS["corpus"], cot="full", seed=seed,
                             capture=dict(layers=CAPTURE_LAYERS, plan=False, semantic=True, narEvals=NAR_EVALS)))
            for ax, poles in AXES.items():
                for pi, pole in enumerate(poles):
                    runs.append(dict(name=f"{g}_{ax}{'+-'[pi]}_s{seed}", style=f"{base}, {pole}",
                                     lyrics=LYRICS["corpus"], cot="full", seed=seed,
                                     capture=dict(layers=CAPTURE_LAYERS, plan=False, semantic=True,
                                                  narEvals=NAR_EVALS)))
    write("exp2_corpus", dict(out="corpus/exp2", arWeights="int8", vae=True, runs=runs))


def _nar_avg(axis, layer):
    """The NAR direction averaged over the captured evaluations (native NAR
    steering adds one vector at every evaluation)."""
    import numpy as np
    vs = [np.fromfile(ROOT / f"cache/dirs/{axis}_synthesize_L{layer}_e{e}.f32", dtype=np.float32) for e in NAR_EVALS]
    p = ROOT / f"cache/dirs/{axis}_nar_L{layer}_avg.f32"
    np.mean(vs, axis=0).astype(np.float32).tofile(p)
    return f"cache/dirs/{axis}_nar_L{layer}_avg.f32"


def exp2c(axis="energy", ar_layers="9,15,21", nar_layers="9,15,21", ar_scales="1,2,3", nar_scales="2,4,8,16",
          max_tokens="0"):
    """Steering verification on held-out prompts: base renders, then the
    semantic stage steered (plan fixed: plan=false) and the NAR steered on the
    base's own tokens (synthesize mode), at increasing scale, in both signs.

    Energy writes runs/exp2c (jobs/exp2c.json); any other axis runs/exp2c_<axis>
    (jobs/exp2c_<axis>.json), with its own base renders (the same seeds). An
    empty layer list skips that seam. max_tokens > 0 caps the semantic stage
    (a steer that never ends the song hits the cap: steer_eval counts it broken).

        make_jobs.py exp2c density 21,24 "" 1,2,3 "" 4000
    """
    from prompts import HELDOUT, LYRICS
    ints = lambda s: [int(x) for x in s.split(",") if x]
    floats = lambda s: [float(x) for x in s.split(",") if x]
    ar_layers, nar_layers = ints(ar_layers), ints(nar_layers)
    ar_scales, nar_scales = floats(ar_scales), floats(nar_scales)
    out = "runs/exp2c" if axis == "energy" else f"runs/exp2c_{axis}"
    cap = dict(semanticSampling=dict(maxTokens=int(max_tokens))) if int(max_tokens) > 0 else {}
    base, steer = [], []
    for g, style in HELDOUT.items():
        for seed in (21, 22):
            b = f"{g}_base_s{seed}"
            base.append(dict(name=b, style=style, lyrics=LYRICS["corpus"], cot="full", seed=seed, **cap))
            for sign in (1, -1):
                for s in sorted(set(ar_scales) | set(nar_scales)):
                    sc = sign * s
                    tag = f"{'+' if sign > 0 else '-'}{s:g}"
                    for L in (ar_layers if s in ar_scales else []):
                        steer.append(dict(name=f"{g}_ar{L}{tag}_s{seed}", style=style, lyrics=LYRICS["corpus"],
                                          cot="full", seed=seed, **cap,
                                          steer=[dict(layer=L, nar=False, plan=False, semantic=True, scale=sc,
                                                      delta=f"cache/dirs/{axis}_semantic_L{L}_e-1.f32")]))
                    for L in (nar_layers if s in nar_scales else []):
                        steer.append(dict(name=f"{g}_nar{L}{tag}_s{seed}", mode="synthesize",
                                          **{"from": f"{out}/{b}"},
                                          steer=[dict(layer=L, nar=True, scale=sc, delta=_nar_avg(axis, L))]))
    write("exp2c" if axis == "energy" else f"exp2c_{axis}",
          dict(out=out, arWeights="int8", vae=True, runs=base + steer))


MEMORIES = (125, 250, 500, 1000, 2000)


def exp5():
    """Endless streams (native/endless.js): 10 s chunks, each opened by the
    last `memory` tokens. Constant styles for coherence and seams, a tour of
    four styles for how fast a style change is heard, references per style,
    a 6-minute run for drift, and sung streams for the lyrics."""
    from prompts import STREAM_STYLES
    tour = [["folk", 4], ["synth", 4], ["metal", 4], ["lofi", 4], ["folk", 4]]
    streams = []
    for m in MEMORIES:
        for key in ("folk", "synth"):
            streams.append(dict(name=f"const_{key}_m{m}", memory=m, seed=1, schedule=[[key, 18]]))
        for seed in (1, 2):
            streams.append(dict(name=f"tour_m{m}_s{seed}", memory=m, seed=seed, schedule=tour))
    for key in ("folk", "synth", "metal", "lofi"):
        for seed in (11, 12):
            streams.append(dict(name=f"ref_{key}_s{seed}", memory=500, seed=seed, schedule=[[key, 12]]))
    streams.append(dict(name="long_folk_m500", memory=500, seed=3, schedule=[["folk", 36]]))
    for m in (500, 1000):
        streams.append(dict(name=f"voice_m{m}", memory=m, seed=1, schedule=[["voice", 18]]))
    write("exp5", dict(out="runs/exp5", arWeights="int8", styles=STREAM_STYLES, streams=streams))


SCORE_STYLES = {
    "orig": "{base}",
    "synth": "synthwave, synth, drum machine, {voice}, {bpm} bpm",
    "jazz": "jazz-funk, Rhodes piano, groovy bass, {voice}, {bpm} bpm",
    "metal": "heavy metal, distorted electric guitar, {voice}, {bpm} bpm",
    "orch": "orchestral, strings, piano, {voice}, {bpm} bpm",
}


def exp6():
    """One score, many styles: exp1's folk and rock scores (cot full, the
    ABC supplied, so the plan stage is skipped) rendered in five styles x two
    seeds, for beat alignment and harmony across renders (exp6.py, after
    separation)."""
    from prompts import LYRICS
    src = {"folk": ("runs/exp1/folk_gen_W0", LYRICS["river"], "female vocal", 92),
           "rock": ("runs/exp1/rock_gen_W0", LYRICS["neon"], "powerful vocal", 128)}
    runs = []
    for song, (d, lyrics, voice, bpm) in src.items():
        pj = json.loads((ROOT / d / "plan.json").read_text(encoding="utf-8"))
        for key, fmt in SCORE_STYLES.items():
            style = fmt.format(base=pj["style"], voice=voice, bpm=bpm)
            for seed in (1, 2):
                runs.append(dict(name=f"{song}_{key}_s{seed}", style=style, lyrics=lyrics, cot="full",
                                 abc=pj["abc"], seed=seed))
    write("exp6", dict(out="runs/exp6", arWeights="int8", vae=True, runs=runs))


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])

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


def exp5b():
    """Exp 5 showed a style change barely moves a continuing stream (the
    history wins even at 5 s). Two levers on the same tour: CFG (the negative
    is the same history with no tags, so the scale amplifies what the new tags
    add), and a bridge (a fresh clip of the new style, and splice's transition
    into it from the old music)."""
    from prompts import STREAM_STYLES
    tour = [["folk", 4], ["synth", 4], ["metal", 4], ["lofi", 4], ["folk", 4]]
    streams = []
    for m in (250, 500):
        for seed in (1, 2):
            for g in (1.5, 2.0, 3.0):
                streams.append(dict(name=f"tour_m{m}_cfg{g:g}_s{seed}", memory=m, seed=seed, cfgScale=g, schedule=tour))
            streams.append(dict(name=f"tour_m{m}_bridge_s{seed}", memory=m, seed=seed, bridge=True, schedule=tour))
    for seed in (1, 2):
        streams.append(dict(name=f"tour_m250_bridge_cfg2_s{seed}", memory=250, seed=seed, bridge=True, cfgScale=2.0,
                            schedule=tour))
    for g in (2.0, 3.0):
        streams.append(dict(name=f"const_folk_m500_cfg{g:g}", memory=500, seed=1, cfgScale=g, schedule=[["folk", 18]]))
    write("exp5b", dict(out="runs/exp5", arWeights="int8", styles=STREAM_STYLES, streams=streams))


def exp5c():
    """An endless stream under a score: every chunk's prefix carries exp1's
    folk score (cot full, the ABC supplied) and its lyrics, the style text
    changes on the tour. Does the score's tempo hold across chunks and style
    changes, and does a change move the sound?"""
    from prompts import LYRICS
    pj = json.loads((ROOT / "runs/exp1/folk_gen_W0/plan.json").read_text(encoding="utf-8"))
    styles = {}
    for key, fmt in SCORE_STYLES.items():
        if key == "orig":
            continue
        styles["score_" + key] = dict(style=fmt.format(base="", voice="female vocal", bpm=92), lyrics=LYRICS["river"],
                                      abc=pj["abc"])
    tour = [["score_synth", 4], ["score_metal", 4], ["score_jazz", 4], ["score_orch", 4]]
    streams = []
    for m in (250, 500):
        streams.append(dict(name=f"score_tour_m{m}", memory=m, seed=1, cot="full", schedule=tour))
        streams.append(dict(name=f"score_tour_m{m}_bridge", memory=m, seed=1, cot="full", bridge=True, schedule=tour))
    streams.append(dict(name="score_const_m500", memory=500, seed=1, cot="full", schedule=[["score_synth", 18]]))
    write("exp5c", dict(out="runs/exp5", arWeights="int8", styles=styles, streams=streams))


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


def exp7():
    """The open questions from exp5: (a) the bridge moves the style but drops
    the beat at the fresh clip: does opening the change chunk with only the
    last 0.5-2 s of the old music (changeMemory) keep the beat and still move?
    (b) scored references, so a scored tour's chunks can be labelled against
    styles under the same score, plus drop and CFG levers under the score;
    (c) rolling lyrics: a lyric window that advances each chunk, against one
    long sheet."""
    from prompts import LYRICS, STREAM_STYLES
    tour = [["folk", 4], ["synth", 4], ["metal", 4], ["lofi", 4], ["folk", 4]]
    styles = dict(STREAM_STYLES)
    styles["voice_long"] = dict(style=STREAM_STYLES["voice"]["style"], lyrics=LYRICS["long"])
    pj = json.loads((ROOT / "runs/exp1/folk_gen_W0" / "plan.json").read_text(encoding="utf-8"))
    skeys = [k for k in SCORE_STYLES if k != "orig"]
    for key in skeys:
        styles["score_" + key] = dict(style=SCORE_STYLES[key].format(base="", voice="female vocal", bpm=92),
                                      lyrics=LYRICS["river"], abc=pj["abc"])
    stour = [["score_synth", 4], ["score_metal", 4], ["score_jazz", 4], ["score_orch", 4]]
    streams = []
    for seed in (1, 2):
        for m in (12, 25, 50):
            streams.append(dict(name=f"tour_m250_drop{m}_s{seed}", memory=250, seed=seed, changeMemory=m, schedule=tour))
        for key in skeys:
            streams.append(dict(name=f"scoref_{key}_s{seed}", memory=250, seed=10 + seed, cot="full",
                                schedule=[["score_" + key, 8]]))
        for m in (25, 50):
            streams.append(dict(name=f"stour_m250_drop{m}_s{seed}", memory=250, seed=seed, cot="full", changeMemory=m,
                                schedule=stour))
        streams.append(dict(name=f"stour_m250_bridge_s{seed}", memory=250, seed=seed, cot="full", bridge=True,
                            schedule=stour))
        streams.append(dict(name=f"stour_m250_s{seed}", memory=250, seed=seed, cot="full", schedule=stour))
    streams.append(dict(name="stour_m250_bridge_cfg2_s1", memory=250, seed=1, cot="full", bridge=True, cfgScale=2.0,
                        schedule=stour))
    lines = [l for l in LYRICS["long"].splitlines() if l.strip()]
    for per in (1, 2):
        streams.append(dict(name=f"roll_r{per}_w4", memory=500, seed=1, schedule=[["voice", 24]],
                            lyricsRoll=dict(lines=lines, perChunk=per, window=4)))
    streams.append(dict(name="sheet_long", memory=500, seed=1, schedule=[["voice_long", 24]]))
    write("exp7", dict(out="runs/exp7", arWeights="int8", styles=styles, streams=streams))


def exp7b():
    """Exp 7's follow-ups. Under a score the bridge keeps the beat into the
    bridge chunk but jumps it into the fresh clip, with the tempo shared:
    bridgeTrims cuts the fresh clip n ways inside one beat and keeps the cut
    the bridge leads into best (NLL). And lyrics that advance by what Whisper
    hears in each rendered chunk, not at a fixed rate."""
    from prompts import LYRICS
    styles = json.loads((JOBS / "exp7.json").read_text(encoding="utf-8"))["styles"]
    stour = [["score_synth", 4], ["score_metal", 4], ["score_jazz", 4], ["score_orch", 4]]
    streams = [dict(name="stour_m250_bridge_s3", memory=250, seed=3, cot="full", bridge=True, schedule=stour)]
    for seed in (1, 2, 3):
        for n in (4, 8):
            streams.append(dict(name=f"stour_m250_trim{n}_s{seed}", memory=250, seed=seed, cot="full", bridge=True,
                                bridgeTrims=n, bridgeCandidates=1, trimSpan=17, schedule=stour))
    lines = [l for l in LYRICS["long"].splitlines() if l.strip()]
    for w in (4, 6):
        streams.append(dict(name=f"roll_adapt_w{w}", memory=500, seed=1, schedule=[["voice", 24]],
                            lyricsRoll=dict(lines=lines, perChunk=1, window=w, adaptive=True)))
    write("exp7b", dict(out="runs/exp7", arWeights="int8", styles=styles, streams=streams))


def exp7c():
    """NLL does not see the beat (exp7b: the trims' best was the uncut clip 7
    times in 9). phaseTrim measures it: the stream's beat from its latents,
    the fresh clip's from a probe render, and the cut that lines them up.
    Otherwise exp7's scored bridge (4 candidates), seeds 1-3."""
    styles = json.loads((JOBS / "exp7.json").read_text(encoding="utf-8"))["styles"]
    stour = [["score_synth", 4], ["score_metal", 4], ["score_jazz", 4], ["score_orch", 4]]
    streams = [dict(name=f"stour_m250_phase_s{seed}", memory=250, seed=seed, cot="full", bridge=True, phaseTrim=True,
                    trimSpan=17, beatFrames=16.835, schedule=stour) for seed in (1, 2, 3)]
    write("exp7c", dict(out="runs/exp7", arWeights="int8", styles=styles, streams=streams))


def exp7d():
    """exp7c's beat fit let the period wander (16.5-17.0 frames) and the
    20 s extrapolation to the join turned that into up to 0.4 beats; every
    scored render holds Q:1/4=90 exactly, so the period is fixed at 16.667
    frames and only the phases are measured."""
    styles = json.loads((JOBS / "exp7.json").read_text(encoding="utf-8"))["styles"]
    stour = [["score_synth", 4], ["score_metal", 4], ["score_jazz", 4], ["score_orch", 4]]
    streams = [dict(name=f"stour_m250_phase90_s{seed}", memory=250, seed=seed, cot="full", bridge=True, phaseTrim=True,
                    trimSpan=17, beatFrames=25 * 60 / 90, beatFit=False, schedule=stour) for seed in (1, 2, 3)]
    write("exp7d", dict(out="runs/exp7", arWeights="int8", styles=styles, streams=streams))


def exp7e():
    """exp7d's misses split between the two sides of the fit (the latent
    flux's phase is within 0.1 beats of the audio's in 71% of windows): the
    beat read from the decoded audio instead (85%), seeds 1-5, and the plain
    bridge on seeds 4-5 to match."""
    styles = json.loads((JOBS / "exp7.json").read_text(encoding="utf-8"))["styles"]
    stour = [["score_synth", 4], ["score_metal", 4], ["score_jazz", 4], ["score_orch", 4]]
    streams = [dict(name=f"stour_m250_phaseaudio_s{seed}", memory=250, seed=seed, cot="full", bridge=True,
                    phaseTrim=True, beatSensor="audio", trimSpan=17, beatFrames=25 * 60 / 90, schedule=stour)
               for seed in (1, 2, 3, 4, 5)]
    streams += [dict(name=f"stour_m250_bridge_s{seed}", memory=250, seed=seed, cot="full", bridge=True, schedule=stour)
                for seed in (4, 5)]
    write("exp7e", dict(out="runs/exp7", arWeights="int8", styles=styles, streams=streams))


def exp7f():
    """exp7e's sensors were right on both sides (stream fit within 0.08 beats
    of the audio in 12 of 15, the fresh clip placed within 0.09 of where it
    was heard in 14 of 15), and still 7 of 15 joins missed: the stream moved
    its phase after the window the fit saw (often by half a beat inside the
    last old chunk, which the NAR had not rendered yet). phaseFrom 'bridge'
    reads the beat at the bridge's end, from a probe render, and cuts the
    fresh clip after the bridge is written."""
    styles = json.loads((JOBS / "exp7.json").read_text(encoding="utf-8"))["styles"]
    stour = [["score_synth", 4], ["score_metal", 4], ["score_jazz", 4], ["score_orch", 4]]
    streams = [dict(name=f"stour_m250_phasebridge_s{seed}", memory=250, seed=seed, cot="full", bridge=True,
                    phaseFrom="bridge", trimSpan=17, beatFrames=25 * 60 / 90, schedule=stour)
               for seed in (1, 2, 3, 4, 5)]
    write("exp7f", dict(out="runs/exp7", arWeights="int8", styles=styles, streams=streams))


def exp7g():
    """exp7f's misses: 4 of 5 the bridge's final render put its accents half
    a beat from where its probe render had them (a second NAR draw), 2 the
    energy sensor and librosa disagreeing by half a beat on the same audio.
    keepProbe: the probe render of the bridge is its render."""
    styles = json.loads((JOBS / "exp7.json").read_text(encoding="utf-8"))["styles"]
    stour = [["score_synth", 4], ["score_metal", 4], ["score_jazz", 4], ["score_orch", 4]]
    streams = [dict(name=f"stour_m250_phasekeep_s{seed}", memory=250, seed=seed, cot="full", bridge=True,
                    phaseFrom="bridge", keepProbe=True, trimSpan=17, beatFrames=25 * 60 / 90, schedule=stour)
               for seed in (1, 2, 3, 4, 5, 6, 7)]
    streams += [dict(name=f"stour_m250_bridge_s{seed}", memory=250, seed=seed, cot="full", bridge=True, schedule=stour)
                for seed in (6, 7)]
    write("exp7g", dict(out="runs/exp7", arWeights="int8", styles=styles, streams=streams))


def exp7h():
    """exp7g's misses were all the sensor: the trim's energy flux and the
    judging librosa envelope half a beat apart on the same audio. The beat
    read with spectral flux instead (half a beat off librosa in 4% of 10 s
    windows, energy flux 11%), seeds 1-7, else as exp7g."""
    styles = json.loads((JOBS / "exp7.json").read_text(encoding="utf-8"))["styles"]
    stour = [["score_synth", 4], ["score_metal", 4], ["score_jazz", 4], ["score_orch", 4]]
    streams = [dict(name=f"stour_m250_phasespec_s{seed}", memory=250, seed=seed, cot="full", bridge=True,
                    phaseFrom="bridge", keepProbe=True, beatSensor="spectral", trimSpan=17, beatFrames=25 * 60 / 90,
                    schedule=stour) for seed in range(1, 8)]
    write("exp7h", dict(out="runs/exp7", arWeights="int8", styles=styles, streams=streams))


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])

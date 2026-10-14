# YuE2 research: findings

Provenance (sections 1-6; sections 7-9 use bin2, bro fa2e8a81 + brosoundml ffab953 + brotensor 6012b83):
- Native numbers come from the `bin/` snapshot (bro `87b92fb1`, brosoundml `ea4146a`, Release, INT8 AR) on GPU 0, a 4090 on bus 01.
- Reference numbers come from PyTorch `yue2` (BF16 eager, on bus 01 or GPU 1). Its timings are not latency claims; only the native timings are.
- Songs: folk, rock and synthpop, 170–184 s, cot full.
- The tables are in `tables/`. Generated audio and the corpus are gitignored.

## 1. Streaming: yes, with long enough windows

`narChunkFrames=50` changes the whole song, not just its seams:
- brightness (spectral centroid) drops to ×0.70;
- mel L1 against the whole-song render is 1.6–1.9, where re-rendering with only the noise changed differs by 0.32–0.36 (the natural spread).

Windows of 500 frames (20 s), with 25 frames of inpainted left context and 25 of lookahead, match whole-song quality:
- mel L1 0.27–0.34;
- no step at the seams (flux z ≈ 0);
- centroid ×0.95–0.97.

Starting with a 250-frame window and then using 500 is close behind (mel L1 0.31–0.46). More left context, the full song history or absolute positions don't rescue short windows; what matters is the size of each solved window.

Tables: `exp1_seams.md`, `exp1c_seams.md`.

## 2. A realtime radio is easy

Native stage speeds:

| stage | speed |
|---|---|
| plan | 2.4 s |
| semantic | 350 tok/s (14× realtime) |
| whole-song NAR | 0.09 s per audio second |
| VAE | 6.2 s per song |

About 36 s of GPU time builds a 170 s song, so prefetching one song is enough, at about 22% GPU duty.

Time to first audio with growing windows is about 4.3 s with cot full, or about 1.9 s without the plan.

Running two stages in separate processes on one GPU just time-slices: 28.4 s concurrent against 25.3 s back to back. Interleave the stages in one process instead.

Tables: `exp1_timing.md`, `latency_native.json`.

## 3. Attributes

- Energy, density, tempo-feel and brightness share one "arousal" direction at semantic layer 9 (cosine ≈ 0.95 between the axes). They separate by layer 21.
- Energy is cleanly linear:
  - semantic layers 21–24: d' 3.8, held-out-genre sign accuracy 1.0;
  - NAR: d' 2.5–3.8;
  - latent mean: d' 2.5.
- Tempo lives only in the plan's `Q:` header. Nothing changes it mid-song, a plan edit included: the beat stayed at 89 and 129 bpm.
- "Vocal / instrumental" has no audio effect when lyrics are present.
- Whole-song steering is clean in the calm direction down to −2 at every seam.
- In the energetic direction, spectral flatness rises 7–9× at +1 for semantic-layer steers (AR L9 and L21) and 3.4× for the latent offset. That may be a noisy wash; it needs listening.
- Past the cap:
  - AR steers at |scale| ≥ 3 never end the song (they hit 9000 tokens);
  - a latent offset of 4, and the NAR L21 steer at 4 or more, break the audio.

- Style axes at semantic layer 24 (whole-song steer, 4 held-out prompts, scales ±1–3; cosine with energy 0.41–0.56):
  - density: + is a full band (crest −6 dB at +1, 4/4 clips), − is sparse;
  - timbre: + is bright (centroid +1.3 kHz at +1, 4/4), − is warm and dark;
  - tempo feel: no scale raised the onset rate (0–1 of 4 at every positive scale). It acts like an energy steer, so tempo stays a plan (`Q:`) control.
  - Like energy, density and timbre are clean to −1, lose one clip in four at −2, and flatten the spectrum about 6× at +1. Layers 21 and 24 behave alike.
  - Shipped as stream knobs `density` and `timbre` (brosoundml, [−1.5, 1]); `export_dirs.py` writes them into `yue2_style_directions.inc`.

Tables: `exp2_prompt.md`, `exp2_seams.md`, `exp2c_steer_energy.md`, `exp2c_steer_{density,timbre,tempo}.md`.

## 4. Interrupt

Each run switches at 60 s and re-renders a 10 s span with the design window. An onset step at the switch above about 2 is audible.

| mechanism | native latency | effect at a safe scale | step at the switch |
|---|---|---|---|
| latent offset | ≈ 0.05 s | scale 0.5: +1.1–1.5 LUFS, +0.5 kHz; scale 1: +2.2–3.3 LUFS, +0.9–1.0 kHz | 0.02–0.21 |
| NAR L21 steer | ≈ 1–1.5 s | scale 2: energy +0.45–0.53 | 0.8–1.4 |
| AR L21 steer on the continuation | ≈ 2–2.5 s | scale 1: energy +0.42–0.58 on folk and rock; inconsistent on synthpop | below 1 |

- The latent offset is clean, but it only makes the song louder and brighter.
- The NAR L21 steer breaks at scale 4 or more (step 2.6–3.7). At 8–16 it adds 11–16 LUFS and clips.
- NAR L9 is a percussive knob: up to +3.2 onsets per second at scale 8.
- The AR L21 steer is a real arrangement change.
- Re-prompting the style, re-prompting the plan's tempo or labels, or re-rolling the continuation all do about nothing mid-song: the song's own history dominates.

- The style axes on a continuation (AR L24, scales −1.5, −1, 0.5, 1, folk and rock; `scripts/queue_style.sh`): every step at the switch is below 1 (−1.3 to +0.73, with the unsteered continuation at −1.3 to +0.18). Mid-song they do less than over a whole song, as every steer does:
  - density +1: centroid +0.4–0.8 kHz, flatness 2–2.5×;
  - timbre: inconsistent mid-song. Folk moved +0.8 kHz at 0.5 and not at 1; rock moved +0.6 kHz at 1.
  - tempo: as in exp2c, no onset-rate change.

Tables: `exp3_interrupt.md`, `exp3_{density,timbre,tempo}_interrupt.md`.

## 5. VRAM: why 14 GB, and about 5 GB to recover

Nothing in the current setup is silly. Peak for a default INT8 song:

| component | GB |
|---|---|
| CUDA context + Vulkan + engine | 1.50 |
| VAE | 0.50 |
| BF16 weights | 7.22 |
| INT8 copies, kept beside the BF16 | 1.35 |
| semantic KV, preallocated for prefix + 9000 tokens (112 KB per token) | ≈ 1.23 |
| NAR's copy of the whole song's AR K/V | ≈ 1.17 |
| workspace | ≈ 0.3 |
| **peak** | **13.30** |

CFG is off by default, so it isn't a second branch; with cfgScale 1.5 the peak is 14.46.

Savings:

| saving | GB | cost |
|---|---|---|
| drop the BF16 originals the INT8 copies shadow (prefill with INT8) | ≈ 2.7 | needs a parity check |
| NAR reads the session's K/V in place instead of copying it | ≈ 1.1 | none |
| 500-frame NAR windows | 1.1 (13.30 → 12.21) | none |
| grow the KV on demand | 0.5–0.75 | none |
| BF16 VAE | ≈ 0.25 | to be checked |

Together that is about 8 GB for a song.

Tables: `vram/*.json`.

## 6. Sliding AR context and plan streaming

A sliding AR context window isn't worth it. Scored by mean NLL under the full model:

| config | mean NLL | song ends? |
|---|---|---|
| full context | 3.52 | yes |
| last 2000 tokens kept | 3.47 | yes |
| last 1000 tokens | 3.71 | yes, but the last third drifts |
| last 500 tokens | 3.80 | yes, runs longer |
| last 250 tokens | 3.99 | yes, runs longer |
| re-based positions | 3.76–4.21 | never (6000-token cap) |

A whole song's KV is only 0.47 GB, so there is little to save anyway.

The plan can be built incrementally. Cut after four sections and then extended under the full lyrics, it keeps the same header and finishes every section. Its chords agree with an independent full plan at Jaccard 0.80–0.83.

The plan sits in the prefix ahead of every codec token, though, so editing it rewrites the song from 0.5 s on. Plan edits are a next-song control.

Tables: `exp4_slide_folk.json`, `exp4b_plan.json`, `exp4c_plan_edits.md`.

## 7. An endless stream of 10 s chunks

No songs: one token history, and every 10 s chunk is a `splice()` opened by the last N tokens of it (positions re-based each chunk, cot off), rendered by `inpaint()` with up to 475 solved frames kept on the left and 25 frames of lookahead solved and dropped (`native/endless.js`, bin2 = bro fa2e8a81). Style references are fresh streams per style; "heard as" is a chunk's nearest reference centroid in CLAP space (the references' own chunks: 91–100% right). Tables: `exp5_endless.md`.

Cost and seams:
- A 10 s chunk costs 0.76–0.93 s of semantic sampling (growing with memory), 1.7–1.9 s of NAR and 0.3 s of decode: **27–30% of the GPU**. CFG (two branches, batched) adds almost nothing (0.82 s).
- Joins are inaudible by the flux test: seam z ≈ 0 against the stream's own flux, 0–3% of joins above its 95th percentile. The beat carries across a join as well as across a mid-chunk point (beat-grid onset ratio 2–3 both ways).
- No chunk left the corpus latent range (6.5–9.5), in 900+ chunks.

Memory (how much history each chunk sees):

| memory | constant style, 3 min | 6 min |
|---|---|---|
| 125–1000 tokens (5–40 s) | stays itself (no chunk heard as another style) | 500: steady, CLAP to its reference 0.82 first third, 0.84 last |
| 2000 (80 s) | **ends**: at ~110 s it plays an ending, then near-silence (−32 LUFS, −inf) | |

With 80 s behind it the model hears a song that has run its course. 5–40 s of memory never ends.

Tempo: with no score the "N bpm" in the style text is not held (folk asked at 92 came out 63/96/185 or 117; synth asked at 118, 161).

**A style change barely moves a continuing stream.** On a tour of folk, synth, metal, lo-fi (40 s each), swapping the style text:

| lever | new style heard (share of its chunks) | changes reached within 40 s |
|---|---|---|
| plain, memory 125 / 250 / 500 / 1000 / 2000 | 32 / 22 / 7 / 0 / 0% | 43 / 38 / 29 / 0 / 0% |
| CFG 1.5 / 2 / 3 (the negative is the same history without the tags) | 19–41% | 25–62%, erratic, sometimes the wrong style |
| **bridge** (below), memory 250 / 500 | **75 / 69%** | **100%** |

Folk to synth never arrived plainly at any memory. The history wins even at 5 s, so a style change needs structure, not a stronger prompt:

**The bridge.** At a change, sample a fresh 10 s clip of the new style (no history, 0.74 s), then `splice()` a 10 s transition from the old music into it (4 candidates scored by how well the fresh clip follows, 3.4 s), and carry on from the fresh clip. Every one of 16 changes was heard as the new style from the fresh clip on; the transition chunk is still the old style. The cost: the join into the fresh clip has the stream's only rhythm break. The beat carries into the bridge (onset ratio 3.65) but not out of it (1.29, random phase): the fresh clip comes in at its own tempo and phase.

## 8. A score holds the tempo, in songs and in streams

One score (exp1's folk and rock, cot full with the ABC supplied) in five styles x two seeds (`exp6.py`, tables `exp6_shared_score.md`):
- Every render keeps the score's tempo: folk 89.1 bpm against `Q:1/4=90`, rock 129.2 against `Q:1/4=65` (double time). (89.1 is the tempogram's frame rounding: exp7's phase measurements show the folk score held at exactly 90, a beat of 16.667 frames.)
- Two renders differ by an **offset** (the intro's length), not a drift. After one shift (the median lag of a windowed cross-correlation of the vocal stems' onsets), 88% of folk pairs line up within 50 ms. 6 of 10 rock renders line up within 25 ms of each other; the other 4 drift somewhere mid-song, so a render needs an alignment check before it is used as a layer.
- **Harmony carries:** a vocal against another style's backing of the same score is as consonant as against its own (chroma cosine over sung frames 0.525 against 0.526; another song's backing 0.388).

An endless stream under a score (exp5c: every chunk's prefix carries the folk score, the style text tours synth, metal, jazz, orchestral): the tempogram of every stream peaks at 89 in both halves, through style changes and bridges. A change moves the sound, though less than without a score (CLAP metal-minus-synth from −0.37 to about −0.15, against a flat −0.35 constant). A fresh clip under a score starts from the top of the score.

## 9. Stems: separation, not the model

YuE2 has no stems: one mixed token per frame and one latent. The score is two-voice (`V: Vocal`, `V: Ins`, chords), so the vocal line is separate symbolically.

htdemucs_ft (MIT code and weights; `sep.py`) on YuE2's output:
- **37 ms per audio second** on a 4090 (27x realtime): a 10 s chunk in 0.37 s.
- The vocal stem is clean: Whisper large-v3 hears 73 words a song on it, 94% of them the lyrics, against 28 words (79%) on the mix.
- Instrumental streams leave −8 to −23 dB in the vocal stem (metal −60), all wordless (Whisper: "♪" only): vocalise or leakage, not lyrics.

A sung endless stream (the lyrics in every chunk's prefix, memory 500) walks the lyric sheet roughly in order (verse, pre-chorus, chorus, second verse by 100 s), then past the sheet it makes up lines ("We are the one", "Tomorrow starts through").

## 10. Style changes that keep the beat; lyrics that keep going

`native/endless.js` options, `exp7_eval.py`, tables `exp7_stream.md`. Scored tours (the folk score; synth, metal, jazz, orchestral, 40 s each, memory 250) are labelled against scored references (the same score, one style each; their chunks heard as their own style 93–100%). The beat across a join is measured as a phase: the comb phase at the score's period (exactly 2/3 s) of the 6 s before and the 6 s after, their distance in beats (0 = the beat goes on, 0.5 = the off-beat, random median 0.25). Inside a stream's own chunks the phase holds (median 0.02).

**Memory drop.** The first chunk of a new style opened by only the last few tokens of the old music (`changeMemory`), the rest as usual:

| lever | unscored: new style heard / changes reached | scored: heard / reached | scored: phase into the change |
|---|---|---|---|
| plain (memory 250) | 22% / 38% | 17% / 33% | 0.00 |
| drop to 12 tokens (0.5 s) | **78% / 100%** | | |
| drop to 25 (1 s) | 57% / 71% | 62% / 83% | 0.06 (2 of 6 jumped) |
| drop to 50 (2 s) | 31% / 50% | 25% / 50% | 0.02 |
| bridge | 75% / 100% | 42% / 84% | 0.00, then 0.31 into the fresh clip |

Unscored, a 12-token drop moves the style as far as the bridge, for nothing extra (no fresh clip, no 4-candidate transition). The unscored styles each have their own tempo, so the beat changes with the style anyway. Under a score the trade is plain: the more memory, the less it moves, and the beat holds best at 2 s.

**The bridge's beat, under a score.** The tempo is shared, so the beat into the fresh clip is a phase to set, at most one beat (17 frames) of cut from its start. In order:
- Cutting the clip several ways and keeping the cut the bridge's NLL likes best does nothing: the uncut clip wins 7 of 9 times. The model's likelihood does not see the phase.
- Measuring it: the stream's beat and a probe render of the fresh clip, then the cut that lines them up. Its error came from three places, each fixed in turn. (1) A fitted period (16.5–17.0 frames) extrapolated 20 s: fixed by the score's own, 16.667. (2) The stream read where the NAR had rendered it, a chunk behind; the last old chunk moved its phase (often by half a beat) as often as not: read at the bridge's end instead, from a probe render of the bridge. (3) The bridge's final render, a second NAR draw, put its accents half a beat from the probe's in 4 of 15: the probe render kept as the render.
- The beat sensor: latent flux agrees with librosa's onset envelope (within 0.1 beats) in 69% of windows, an energy flux of the decoded audio in 82%, spectral flux (48 log bands, 100 fps, a 2048-point FFT in JS) in 90%, half a beat off in 4%.

| into the fresh clip (21 bridges, 7 seeds) | within 0.11 beats | within 0.25 | median |
|---|---|---|---|
| plain bridge | 5 | 8 | 0.31 |
| measured, energy flux, read at the bridge's end, probe kept | 13 | 14 | 0.08 |
| **measured, spectral flux, read at the bridge's end, probe kept** | **15** | **20** | **0.05** |

The style still moves as with the plain bridge (43% of chunks, 78% of changes reached, against 42% and 84%), and the measurement adds about 1 s per bridge (the fresh clip's probe render and decode; the bridge's probe is its render).

**Lyrics.** A sung stream (memory 500, 24 chunks = 4 minutes) against a 40-line sheet with no line twice, transcribed per 10 s window from the separated vocal (Whisper large-v3; a line counts as sung when 60% of its words are heard):

| lyrics in each chunk's prefix | distinct lines sung | furthest line | lines sung again later | words not in the sheet | sung lines inside the chunk's window |
|---|---|---|---|---|---|
| the whole sheet | 12 | 24 | **24** (it loops over lines 0–11) | 3% | |
| a 4-line window, 1 line further each chunk | 12 | 21 | 1 | 5% | 75% |
| a 4-line window, 2 lines further each chunk | 15 | 39 | 1 | 24% | 88% |
| **a 4-line window after the last line heard** | **24** | 24 | 4 | 17% | 87% |
| a 6-line window after the last line heard | 18 | 24 | 7 | 9% | 88% |

Given the whole sheet, the stream sings its first verse, pre-chorus and chorus over and over. A window moved at a fixed rate either lags (one line a chunk: it sings slower than that, then sticks) or skips (two). Moving it past the last line Whisper heard in the rendered chunks (one chunk behind, 0.4 s of Whisper on the mix per chunk; one line on if nothing moved in 3 chunks) sings twice as many lines as the whole sheet, in order.

## Continuous streaming: what this means for the design

- **The stream is chunks, not songs.** 10 s chunks with 5–40 s of memory are seamless, never end, and cost under a third of the GPU, so two or three branches can be built ahead of a listener.
- **Moving through styles is a structural move, not a prompt edit.** Swapping the text does not move the music. Without a score, dropping the memory to 0.5 s at the change moves it as reliably as a bridge, for free. Under a score, a bridge (a fresh clip plus a spliced transition) reaches 84% of changes within 40 s; a 1 s memory drop moves more chunks (62%) but jumps the beat a third of the time.
- **The beat across a bridge is fixable under a score.** The score fixes the tempo; the phase is set by cutting the fresh clip's start at the beat read off the bridge's own render (spectral flux), 71% of joins within 0.11 beats and 95% within 0.25 (plain: 24% and 38%), about 1 s per bridge.
- **A vocal layer over a moving backing is feasible.** Same score (tempo and chords), the vocal separated from one render and the backing from another, one shift aligns most pairs; 27x realtime separation keeps it live. Demos for listening: `runs/exp6/demos/*_backing.flac` (folk_orig_s1's and rock_orig_s1's vocal over each other render's backing, shifted).
- **Lyrics roll.** The prefix carries a few lines at a time, moved on by what Whisper hears the stream sing; the whole sheet makes it loop.
- **Not measured yet:** the measured cut by ear (the numbers are librosa's onset envelope; a residual 0.2-beat join may or may not be audible), the drop and the trimmed bridge combined, scores other than the folk one, and an adaptive lyric window in a scored stream (the score's melody is written for the whole sheet).

## Recommended design

**Streaming (brosoundml).** One `Yue2Stream` per song, in one process on one device:
- a plan (whole, or a few sections ahead);
- a live semantic session whose KV grows on demand;
- NAR windows of 250 frames, then 500, with 25 frames inpainted on the left and 25 of lookahead, reading the session's K/V in place;
- the existing streaming VAE.

**Radio.**
- Build the next song whole while the current one plays.
- The next song's prompt carries the listener's last steer, plus any tempo or key plan edits.
- Equal-power crossfade of 2–4 s at the outro.
- A skip starts a fresh stream (about 4.3 s to first audio).

**Interrupt, in two stages.**
1. Instantly: a latent offset of at most 1 with a 0.5 s ramp, re-decoded.
2. Within about 2.5 s: an AR L21-steered continuation from the playing frame + 1 s, inpainted and crossfaded in at a window edge. Then drop the offset.

The NAR L21 steer at ≤ 2 (or NAR L9 for percussion) is a texture knob.

**Lab controls:**
- an energy/brightness knob and a density/percussive knob;
- a plan panel for the next song;
- window and ODE-step controls (16 steps halve the NAR cost);
- the radio queue;
- meters with a break warning.

**Size:** about 3.9k LOC.

| area | part | LOC |
|---|---|---|
| brosoundml (≈ 2.3k) | inpainted windows + in-place K/V | 550 |
| | semantic continuation + KV growth | 400 |
| | stream scheduler | 450 |
| | section-ahead plan | 200 |
| | VRAM fixes | 250 |
| | bug guards | 80 |
| | JS binding (`src/api/native_soundml_music.cpp`) | 400 |
| bro (≈ 0.45k) | `docs/music-api.js` | 200 |
| | tests | 250 |
| yue2-lab (≈ 1.15k) | radio | 300 |
| | crossfade player | 250 |
| | interrupt UI | 250 |
| | plan panel | 200 |
| | meters | 150 |

## Bugs found (not fixed here)

- **B1. CUDA illegal memory access under a strong AR steer.** The error is `CUDA error 700 ... brotensor\src\cuda\tensor.cu:150 in cudaMemcpy D2H`, and the process dies.
  - Repro: job `soul_ar9-8_s21` in `jobs/exp2c.json`: style "soul, female vocal, Rhodes piano, bass", the corpus lyrics, cot full, seed 21, INT8, steer {layer 9, semantic only, scale −8, `cache/dirs/energy_semantic_L9_e-1.f32`}.
  - Likely cause: non-finite logits reach sampling and produce an out-of-range token index.
- **B2. An AR steer at |scale| ≥ 3 never ends the song.** It runs to 9000 tokens. A stream needs a cap tied to the plan's length, or an end bias.
- **B3. `require('./x')` in a `bro-headless <app> script.js` script resolves against the cwd**, not the script's directory, although `docs/brokit-api.js` says it resolves against the calling file. (Fixed in bro 6e328841; bin2's scripts require `./lib.js`.)
- **B4. brotensor's spatial guards (3776145) broke CLAP and Whisper.** They demanded X be exactly (N, C*H*W), and callers hand the same storage over as one row or as a matrix being transposed: every `embedAudio` and `loadWhisper` threw "nchw_to_sequence: shape mismatch". Fixed: the guards check the element count (brotensor 6012b83), and CLAP reshapes its mel image (brosoundml ffab953).

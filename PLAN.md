# yue2-research — plan

Three questions, each with a decision criterion stated before the run.
Everything is measured on the native path where it can be (it is what would
ship), and on the reference model only where the native seams do not reach.

## Meters

- **Seams.** Against the whole-song NAR render of the *same* semantic tokens
  and the *same* noise: latent cosine and relative error by distance from a
  seam; latent frame-to-frame jump at seams / the median jump; log-mel L1 near
  seams vs mid-window; excess spectral flux at the seam in units of the
  reference's flux sd, and where it sits in the reference's flux distribution.
  **Scale:** the same tokens rendered whole-song with *other noise* (`W0n2`) are
  the natural spread; a window that is no further from the reference than that
  is as good as the protocol. Seam flux has no natural spread to hide in, so it
  is judged on its own: a seam whose flux is above the reference's 95th
  percentile is an audible click / step.
- **Attributes.** LUFS, dynamics (95th–10th percentile RMS dB), centroid,
  flatness, rolloff, onset rate, onset strength, tempo, percussive share,
  harmonic vocal-band share, low-end share (`meter.features`), plus a combined
  energy score (loudness + brightness + onset rate + percussive share).
- **Break detector.** Steering quality is never one scalar. A steered clip is
  "broken" when, versus its unsteered twin, any of: flatness more than doubles
  (noise), onset rate collapses below half or the clip goes near silent
  (LUFS < −40), or the latent norm leaves the corpus range. Corroborated by the
  latent statistics, never by a single meter.

## Exp 1 — Streaming feasibility

1a. **Stage timing (native).** For three ~3-minute songs (folk, synthpop, rock;
cot full, INT8): plan / semantic / NAR / VAE seconds, tok/s, first semantic
token, first latent, first audio, total wall, protocol vs `narChunkFrames` 50.

1b. **Window seams (native).** The protocol run's tokens re-synthesized with
`narChunkFrames` 25 / 50 / 100 / 250 (hard windows: each window sees only its
own codec tokens, latent positions restart at 0), plus the whole song with
other noise as the scale.

1c. **Context and inpainting (reference).** What fixes the seams: codec context
on the left / right of the window (the solve covers the context, only the core
is kept), and inpainting the already-emitted latents of the left context
(clamped to the flow interpolant during the ODE). Grid over window, left and
right context. Right context costs latency (tokens must exist), left context
costs compute.

1d. **Parallel stages.** Does semantic AR overlap with NAR on the same GPU
without slowing either below realtime? (The native streaming mode runs windows
between tokens on one thread.)

**Decision:** realtime radio is possible if (steady state) semantic + NAR + VAE
per audio second < 1 s with margin, the first audio comes within a prefetch
budget, and some (window, context) gives seams within the natural spread.

## Exp 2 — Where musical attributes live

2a. **Corpus.** 4 genres × 5 axes (energy, tempo, density, timbre, vocal) × 2
poles × 2 seeds + neutral, cot full, ~30 s each; residual means captured at 10
AR layers (semantic stage) and 10 NAR layers × 4 evaluations; latents and audio
kept. Check first that the poles move the meters at all (the prompt is the
ceiling of a direction found from it).

2b. **Directions and probes.** Per (stage, layer[, eval]): the paired pole
difference (diff of means over genre × seed pairs), its consistency (mean
cosine between per-pair differences), and leave-one-genre-out probe accuracy.
The same in the 64-dim latent space (mean latent per clip).

2c. **Verification by steering.** The energy direction at the best seams,
steered on held-out prompts (genres not in the corpus) at increasing scale:
AR semantic stage, NAR stage, and a direct latent offset. Meters vs scale; the
break point per seam.

**Decision:** a seam is a control surface if the steered meter moves
monotonically with scale, in the prompt's direction, on held-out prompts,
before the break detector fires.

## Exp 3 — Interrupt latency

With semantic tokens already ahead of the playhead, for each candidate seam:
latency from request to the first changed audio, and the size of the change
(energy score delta vs the unsteered render of the same span), and the seam it
leaves at the switch point.

- **A. Latent offset** on frames not yet decoded (VAE only).
- **B. NAR steer** re-synthesizing only the unplayed window(s) from the same
  tokens.
- **C. Re-prompt stage 1:** continue semantic sampling from the current point
  under an edited style prompt, then NAR + VAE.
- **D. AR steer** on the continuation (same prompt, energy vector).

## Deliverable

`FINDINGS.md`: numbers for 1–3, then the recommended design for the streaming
pipeline, radio mode, the interrupt control, and the lab's control surfaces,
with size estimates in lines of code per repo.

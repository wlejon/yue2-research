// Exp 5: an endless stream of short chunks instead of songs.
//
// The stream is one token history. Chunk k (`chunk` tokens, 25 per second)
// is sampled by splice() under chunk k's style, fed only the last `memory`
// tokens of the history as its opening (positions re-based every chunk: the
// model sees a clip, not a song). Then the NAR renders chunk k by inpaint()
// over a window of the history: up to `narLeft` already-solved frames kept on
// the left, chunk k solved, plus `lookahead` frames of chunk k+1 solved and
// thrown away (the next chunk re-solves them with this one kept). The style
// can change at any chunk: that chunk's plan (cot 'off': the prefix only) is
// the new style's, its opening the old music.
//
// Job (YUE_JOB):
// { "out": "runs/exp5", "arWeights": "int8",
//   "styles": { "folk": { "style": "...", "lyrics": "..." }, ... },
//   "streams": [ { "name", "memory": 500, "chunk": 250, "lookahead": 25, "narLeft": 475,
//                  "seed": 1, "odeSteps": 32, "cot": "off",
//                  "schedule": [ ["folk", 4], ["synth", 4], ... ] } ] }
// Writes per stream: tokens.i32, latents.f32 (frames x 64), audio.f32
// (planar L then R, 48 kHz), chunks.json (per chunk: style, frames, op
// seconds, latent norm).
// bin2's bro resolves require against this file (bin/'s against the cwd).
const L = require('./lib.js');
const { fs, now, writeBin, writeJson, mkdirp } = L;

const job = L.readJob();
const yue = L.loadYue(job.arWeights);
const vae = L.loadVae();

// Whisper, only for a stream with adaptive lyrics (it listens to each chunk).
let whisper = null, whisperTok = null, whisperPrompt = null;
if (job.streams.some((st) => st.lyricsRoll && st.lyricsRoll.adaptive)) {
    const wd = process.env.WHISPER || 'D:/projects/brosoundml/weights/whisper-large-v3';
    whisper = bro.stt.loadWhisper(wd);
    whisperTok = bro.stt.loadTokenizer({ vocabPath: wd + '/vocab.json', mergesPath: wd + '/merges.txt',
                                         addedTokensPath: wd + '/added_tokens.json' });
    whisperPrompt = whisperTok.buildPrompt('en', 'transcribe', false);
}

function wait(start, desc) {
    let result = null, info = null;
    const h = start((r, i) => { result = r; info = i || {}; });
    L.pumpUntil(() => info !== null, 3600, desc);
    if (info.error) throw new Error(desc + ': ' + info.error);
    if (info.cancelled || !result) throw new Error(desc + ': cancelled');
    return result;
}

const plans = new Map();
function planFor(key, cot, lyrics) {
    const s = job.styles[key];
    if (lyrics === undefined) lyrics = s.lyrics;
    const k = key + '|' + cot + '|' + lyrics;
    if (plans.has(k)) return plans.get(k);
    const t = now();
    // A style with an abc is planned from it (the plan stage skipped): a stream under a score.
    const o = { ...(s.abc ? { abc: s.abc } : {}), style: s.style, lyrics, cot, seed: 1 };
    const r = wait((done) => yue.plan(Object.assign(o, { onDone: done })), 'plan ' + key);
    const p = { plan: r.plan, seconds: now() - t };
    plans.set(k, p);
    return p;
}

// The beat from latents: frame-to-frame latent change (25 fps), its 9-frame
// local mean removed and rectified, follows the audio's onsets (its comb
// phase is within 0.08 beats of the audio onset envelope's, median over 520
// windows of the scored streams). fitBeat searches the period around P0
// (+-3%, 0.2% steps; fixed when spread is 0) and the phase (1/64 beat),
// linear interpolation at fractional frames; phase in beats, relative to
// frame 0 of `lat`.
function latentOnsets(lat, f0, f1) {
    const n = f1 - f0, fl = new Float32Array(n);
    for (let t = 1; t < n; t++) {
        let s = 0;
        for (let d = 0; d < 64; d++) { const v = lat[(f0 + t) * 64 + d] - lat[(f0 + t - 1) * 64 + d]; s += v * v; }
        fl[t] = Math.sqrt(s);
    }
    const out = new Float32Array(n);
    for (let t = 0; t < n; t++) {
        let s = 0, c = 0;
        for (let j = Math.max(0, t - 4); j <= Math.min(n - 1, t + 4); j++) { s += fl[j]; c++; }
        out[t] = Math.max(0, fl[t] - s / c);
    }
    return out;
}
function fitBeat(env, origin, P0, spread) {
    let best = { v: -1 };
    const steps = spread ? 15 : 0;
    for (let j = -steps; j <= steps; j++) {
        const P = P0 * (1 + j * 0.002);
        for (let q = 0; q < 64; q++) {
            // beat positions in env's frames: origin-relative phase q/64
            let x = ((q / 64) * P - origin) % P;
            if (x < 0) x += P;
            let s = 0, c = 0;
            for (; x < env.length - 1; x += P) {
                const i = Math.floor(x), f = x - i;
                s += env[i] * (1 - f) + env[i + 1] * f; c++;
            }
            if (c && s / c > best.v) best = { v: s / c, P, phase: q / 64 };
        }
    }
    return best;
}

// The beat from decoded audio (st.beatSensor 'audio'): mono 48 kHz, 100
// frames a second (hop 480), the log energy of the signal and of its first
// difference (a crude high-pass), each differenced and rectified, summed.
// Against librosa's onset envelope its comb phase is within 0.1 beats in 85%
// of 20 s windows of the scored streams (latent flux: 71%).
function audioOnsets(mono) {
    const hop = 480, n = Math.floor(mono.length / hop), out = new Float32Array(n);
    let p1 = 0, p2 = 0, prev = 0;
    for (let f = 0; f < n; f++) {
        let s1 = 0, s2 = 0;
        for (let i = f * hop; i < (f + 1) * hop; i++) { const v = mono[i]; s1 += v * v; const d = v - prev; s2 += d * d; prev = v; }
        const e1 = Math.log(1e-6 + s1 / hop), e2 = Math.log(1e-6 + s2 / hop);
        out[f] = f ? Math.max(0, e1 - p1) + Math.max(0, e2 - p2) : 0;
        p1 = e1; p2 = e2;
    }
    return out;
}
// st.beatSensor 'spectral': spectral flux. Mono averaged in pairs to 24 kHz,
// 2048-point Hann frames every 240 samples (100 a second), power summed into
// 48 log-spaced bands (60 Hz..11 kHz), log, positive frame difference averaged
// over the bands, frames indexed by their centres. Against librosa's onset
// envelope: within 0.1 beats in 90% of 10 s windows, half a beat off in 4%
// (energy flux: 82% and 11%), on the scored streams.
const FFT_N = 2048;
const fftCos = new Float64Array(FFT_N / 2), fftSin = new Float64Array(FFT_N / 2), hann = new Float64Array(FFT_N);
for (let i = 0; i < FFT_N / 2; i++) { fftCos[i] = Math.cos(2 * Math.PI * i / FFT_N); fftSin[i] = -Math.sin(2 * Math.PI * i / FFT_N); }
for (let i = 0; i < FFT_N; i++) hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (FFT_N - 1));
const bandOf = new Int32Array(FFT_N / 2 + 1).fill(-1);
for (let b = 0; b <= FFT_N / 2; b++) {
    const f = b * 24000 / FFT_N;
    if (f < 60 || f >= 11000) continue;
    bandOf[b] = Math.min(47, Math.floor(48 * Math.log(f / 60) / Math.log(11000 / 60)));
}
function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const step = n / len, half = len >> 1;
        for (let i = 0; i < n; i += len)
            for (let k = 0; k < half; k++) {
                const c = fftCos[k * step], s = fftSin[k * step];
                const a = i + k, b = a + half;
                const xr = re[b] * c - im[b] * s, xi = re[b] * s + im[b] * c;
                re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
            }
    }
}
function spectralOnsets(mono) {
    const m = new Float64Array(mono.length >> 1);
    for (let i = 0; i < m.length; i++) m[i] = 0.5 * (mono[2 * i] + mono[2 * i + 1]);
    const hop = 240, n = Math.max(0, Math.floor((m.length - FFT_N) / hop)), lead = Math.floor(FFT_N / 2 / hop);
    const out = new Float32Array(n + lead);
    const re = new Float64Array(FFT_N), im = new Float64Array(FFT_N);
    let prev = null;
    for (let f = 0; f < n; f++) {
        for (let i = 0; i < FFT_N; i++) { re[i] = m[f * hop + i] * hann[i]; im[i] = 0; }
        fft(re, im);
        const band = new Float64Array(48);
        for (let b = 0; b <= FFT_N / 2; b++) if (bandOf[b] >= 0) band[bandOf[b]] += re[b] * re[b] + im[b] * im[b];
        for (let j = 0; j < 48; j++) band[j] = Math.log(1e-10 + band[j]);
        let s = 0;
        if (prev) for (let j = 0; j < 48; j++) s += Math.max(0, band[j] - prev[j]);
        out[lead + f] = s / 48;
        prev = band;
    }
    return out;
}
function beatOnsets(st, mono) { return st.beatSensor === 'spectral' ? spectralOnsets(mono) : audioOnsets(mono); }

function monoOf(samples, o, len) {
    const n = samples.length >> 1, m = new Float32Array(Math.min(len, n - o));
    for (let i = 0; i < m.length; i++) m[i] = 0.5 * (samples[o + i] + samples[n + o + i]);
    return m;
}

function latentNorm(lat, f0, f1) {
    let s = 0;
    for (let f = f0; f < f1; f++) {
        let n = 0;
        for (let d = 0; d < 64; d++) { const v = lat[f * 64 + d]; n += v * v; }
        s += Math.sqrt(n);
    }
    return f1 > f0 ? s / (f1 - f0) : 0;
}

for (const st of job.streams) {
    const dir = job.out + '/' + st.name;
    if (fs.existsSync(dir + '/chunks.json') && !job.force) { console.log(st.name + ': exists, skipped'); continue; }
    mkdirp(dir);
    const C = st.chunk || 250, M = st.memory, LA = st.lookahead === undefined ? 25 : st.lookahead;
    const NL = st.narLeft === undefined ? 475 : st.narLeft;
    const cot = st.cot || 'off', seed = st.seed || 1, ode = st.odeSteps || 32;
    const order = [];
    for (const [key, n] of st.schedule) for (let i = 0; i < n; i++) order.push(key);
    const K = order.length;
    const rows = order.map((key, k) => ({ k, style: key, f0: k * C, f1: (k + 1) * C }));
    const t0 = now();

    // st.lyricsRoll = { lines: [...], perChunk, window, adaptive }: chunk k's
    // lyric sheet is `window` lines (section tags kept as lines of their own
    // and not counted), instead of the style's whole sheet. Fixed: starting at
    // line k * perChunk. Adaptive: starting after the last line Whisper heard
    // in the chunks rendered so far (chunk k-2 and before: the NAR runs one
    // chunk behind), one line further if it has not moved for 3 chunks.
    // lyricsRoll.sheet names one of job.sheets in place of its own lines.
    const roll = st.lyricsRoll && st.lyricsRoll.sheet ? { ...st.lyricsRoll, lines: job.sheets[st.lyricsRoll.sheet] } : st.lyricsRoll;
    const sung = roll ? roll.lines.map((l, i) => [l, i]).filter(([l]) => !/^\[/.test(l)) : [];
    const words = (t) => (t.toLowerCase().replace(/’/g, "'").match(/[a-z']+/g) || []);
    const sungWords = sung.map(([l]) => words(l));
    let rollAt = 0, rollMoved = 0, lastHeard = -1;
    const lyricsOf = (k) => {
        if (!roll) return job.styles[order[k]].lyrics;
        let a;
        if (roll.adaptive) {
            const next = Math.max(rollAt, lastHeard + 1);
            if (next > rollAt) { rollAt = next; rollMoved = k; } else if (k - rollMoved >= 3) { rollAt++; rollMoved = k; }
            a = rollAt;
        } else a = k * roll.perChunk;
        a = Math.min(a, sung.length - 1);
        const b = Math.min(sung.length, a + roll.window);
        const i0 = sung[a][1], i1 = sung[b - 1][1] + 1;
        let tag = null;
        for (let i = i0; i >= 0 && !tag; i--) if (/^\[/.test(roll.lines[i])) tag = roll.lines[i];
        const body = roll.lines.slice(i0, i1);
        if (tag && body[0] !== tag) body.unshift(tag);
        rows[k].lyricLines = [a, b];
        return body.join('\n').replace(/\n(\[)/g, '\n\n$1') + '\n';
    };

    // NAR: each chunk inpainted over its window as soon as its lookahead
    // exists, the lookahead solved and dropped; then the chunk decoded (halo
    // on both sides) as a stream would, and, with adaptive lyrics, heard.
    let tokens = new Int32Array(0);
    let lat = new Float32Array(0);
    let narDone = 0;
    const chunkMono = [];
    const narChunk = (k) => {
        const T = tokens.length;
        if (lat.length < T * 64) { const nl = new Float32Array(T * 64); nl.set(lat); lat = nl; }
        const s = Object.assign({}, job.styles[order[k]], { lyrics: rows[k].lyrics });
        const p = planFor(order[k], cot, s.lyrics);
        const c0 = k * C, c1 = c0 + C, e1 = Math.min(T, c1 + LA);
        const s0 = Math.max(0, c0 - NL);
        const win = tokens.slice(s0, e1);
        const wl = new Float32Array(win.length * 64);
        wl.set(lat.subarray(s0 * 64, c0 * 64));
        const t = now();
        const r = wait((done) => yue.inpaint(p.plan, win, wl, c0 - s0, e1 - s0, {
            ...(s.abc ? { abc: s.abc } : {}), style: s.style, lyrics: s.lyrics, cot, seed: seed * 1000 + 500 + k, odeSteps: ode,
            onDone: done }), st.name + ' inpaint ' + k);
        rows[k].inpaintSeconds = now() - t;
        rows[k].narWindow = win.length;
        lat.set(r.latents.subarray((c0 - s0) * 64, (c1 - s0) * 64), c0 * 64);
        rows[k].latentNorm = latentNorm(lat, c0, c1);
        const h0 = Math.max(0, c0 - 12), h1 = Math.min(c1 + 12, (k + 1) * C);
        const td = now();
        const d = vae.decode(lat.slice(h0 * 64, h1 * 64));
        rows[k].decodeSeconds = now() - td;
        if (st.beatSensor === 'audio') chunkMono[k] = monoOf(d.samples, (c0 - h0) * 1920, C * 1920);
        if (roll && roll.adaptive) {
            const tw = now();
            const n = d.samples.length >> 1, o = (c0 - h0) * 1920, len = Math.min(C * 1920, n - o);
            const m = Math.floor(len / 3), x = new Float32Array(m);
            for (let i = 0; i < m; i++) {
                let v = 0;
                for (let j = 0; j < 3; j++) v += d.samples[o + 3 * i + j] + d.samples[n + o + 3 * i + j];
                x[i] = v / 6;
            }
            const text = whisperTok.decode(whisper.transcribe({ samples: x, sampleRate: 16000 }, whisperPrompt,
                                                              { maxNewTokens: 120 }), true).trim();
            const bag = new Set(words(text));
            const heard = [];
            sungWords.forEach((l, i) => { if (l.length && l.filter((w) => bag.has(w)).length / l.length >= 0.6) heard.push(i); });
            // the furthest line heard in or just past this chunk's window (a line far behind is a repeat, not progress)
            const [a, b] = rows[k].lyricLines;
            for (const i of heard) if (i >= a - 1 && i < b + 2) lastHeard = Math.max(lastHeard, i);
            rows[k].heardText = text; rows[k].heardLines = heard; rows[k].listenSeconds = now() - tw;
        }
    };
    const pump = (final) => {
        while (narDone < K && (final ? narDone * C < tokens.length : tokens.length >= (narDone + 1) * C + LA))
            narChunk(narDone++);
    };

    // Semantic: chunk after chunk, each opened by the last M tokens. With
    // st.bridge, a style change is a composed transition instead: chunk k+1
    // is a fresh clip of the new style (no history), and chunk k the bridge
    // splice() writes from the old music into it (st.bridgeCandidates drawn,
    // the one the fresh clip follows best kept). st.bridgeTrims = n: the fresh
    // clip is sampled st.trimSpan frames long (one beat) and cut n ways, its
    // first d frames dropped (d spread over the span); each cut gets its own
    // bridge candidates, and the lowest NLL across all of them is kept: under
    // a score the tempo is shared, so the cut picks the fresh clip's phase.
    const cfg = st.cfgScale === undefined ? {} : { cfgScale: st.cfgScale };
    const append = (m) => { const nt = new Int32Array(tokens.length + m.length); nt.set(tokens); nt.set(m, tokens.length); tokens = nt; };
    // st.changeMemory: the first chunk of a new style is opened by only that
    // many tokens (the old music's last beat or two), the rest by M.
    for (let k = 0; k < K; k++) {
        pump(false);
        rows[k].lyrics = lyricsOf(k);
        const s = Object.assign({}, job.styles[order[k]], { lyrics: rows[k].lyrics });
        const p = planFor(order[k], cot, s.lyrics);
        const change = k > 0 && order[k] !== order[k - 1];
        const mem = change && st.changeMemory !== undefined ? st.changeMemory : M;
        const before = tokens.subarray(Math.max(0, tokens.length - mem));
        if (st.bridge && change && k + 1 < K && order[k + 1] === order[k]) {
            const t = now();
            const nT = st.bridgeTrims || 1, span = nT > 1 || st.phaseTrim || st.phaseFrom ? (st.trimSpan || 17) : 0;
            const fresh = wait((done) => yue.splice(p.plan, Object.assign({
                ...(s.abc ? { abc: s.abc } : {}), style: s.style, lyrics: s.lyrics, cot, seed: seed * 1000 + k + 1,
                before: new Int32Array(0), middleFrames: C + span, candidates: 1, continue: false, onDone: done }, cfg)),
                st.name + ' fresh ' + (k + 1));
            const F = fresh.semantic.subarray(fresh.middle0, fresh.middle0 + C + span).slice();
            const t1 = now();
            let best = null;
            const trials = [];
            // st.phaseTrim (under a score: one tempo): measure instead of guess.
            // The stream's beat (period and phase) from its last 20 s of
            // rendered latents, the fresh clip's phase from a probe render of
            // it, and the cut that puts the fresh clip's beat on the stream's
            // beat at the join, (k+1)*C.
            let cuts = null;
            if (st.phaseTrim) {
                const tp = now();
                const P0 = st.beatFrames || 16.835;
                const r1 = narDone * C, r0 = Math.max(0, r1 - 500);
                const probe = wait((done) => yue.inpaint(p.plan, F, new Float32Array(F.length * 64), 0, F.length, {
                    ...(s.abc ? { abc: s.abc } : {}), style: s.style, lyrics: s.lyrics, cot, seed: seed * 1000 + 900 + k,
                    odeSteps: ode, onDone: done }), st.name + ' probe ' + (k + 1));
                let sb, fb;
                if (st.beatSensor === 'audio') {
                    // envelope frames are 1/100 s, latent frames 1/25: the period and origins scaled by 4
                    const ks = [];
                    for (let j = Math.floor(r0 / C); j < narDone; j++) ks.push(chunkMono[j]);
                    const sm = new Float32Array(ks.reduce((x, m) => x + m.length, 0));
                    let o = 0; for (const m of ks) { sm.set(m, o); o += m.length; }
                    const pd = vae.decode(probe.latents.slice(0, F.length * 64));
                    const s4 = fitBeat(audioOnsets(sm), Math.floor(r0 / C) * C * 4, P0 * 4, false);
                    const f4 = fitBeat(audioOnsets(monoOf(pd.samples, 0, F.length * 1920)), 0, P0 * 4, false);
                    sb = { P: P0, phase: s4.phase }; fb = { phase: f4.phase };
                } else {
                    sb = fitBeat(latentOnsets(lat, r0, r1), r0, P0, st.beatFit !== false);
                    fb = fitBeat(latentOnsets(probe.latents, 0, Math.min(F.length, 150)), 0, sb.P, false);
                }
                // stream beats at sb.phase*P + n*P (absolute frames); the fresh clip's at fb.phase*P + m*P (its frames);
                // a cut of d puts its frame i at (k+1)*C + i - d
                let dd = ((k + 1) * C + fb.phase * sb.P - sb.phase * sb.P) % sb.P;
                if (dd < 0) dd += sb.P;
                cuts = [Math.min(span, Math.round(dd))];
                rows[k + 1].phaseFit = { P: sb.P, streamPhase: sb.phase, freshPhase: fb.phase, cut: cuts[0],
                                         rendered: r1, seconds: now() - tp };
            }
            // st.phaseFrom 'bridge': the stream's beat is read where it meets the
            // fresh clip, at the end of the bridge itself. The bridge is spliced
            // toward the uncut clip, the unrendered chunks up to its end are
            // probe-rendered (the last old chunk moves its phase as often as not,
            // and the NAR runs a chunk behind), and the clip is cut after: the
            // bridge leads into a clip that starts up to a beat later than the
            // one it was written toward.
            if (st.phaseFrom === 'bridge') {
                const tp = now();
                const P0 = st.beatFrames || 16.835;
                const r = wait((done) => yue.splice(p.plan, Object.assign({
                    ...(s.abc ? { abc: s.abc } : {}), style: s.style, lyrics: s.lyrics, cot, seed: seed * 1000 + k,
                    before: before.slice(), middleFrames: C, after: F.slice(0, C), candidates: st.bridgeCandidates || 4,
                    scoreFrames: 125, continue: false, onDone: done }, cfg)), st.name + ' bridge ' + k);
                const bridgeTok = r.semantic.subarray(r.middle0, r.middle0 + C).slice();
                const tb = now();
                const r1 = narDone * C, s0 = Math.max(0, r1 - NL);
                const win = new Int32Array(tokens.length - s0 + C + LA);
                win.set(tokens.subarray(s0)); win.set(bridgeTok, tokens.length - s0); win.set(F.subarray(0, LA), tokens.length - s0 + C);
                const wl = new Float32Array(win.length * 64);
                wl.set(lat.subarray(s0 * 64, r1 * 64));
                const pr = wait((done) => yue.inpaint(p.plan, win, wl, r1 - s0, win.length, {
                    ...(s.abc ? { abc: s.abc } : {}), style: s.style, lyrics: s.lyrics, cot, seed: seed * 1000 + 800 + k,
                    odeSteps: ode, onDone: done }), st.name + ' bridge probe ' + k);
                const b0 = k * C - s0;
                const bd = vae.decode(pr.latents.slice(b0 * 64, (b0 + C) * 64));
                const sb = fitBeat(beatOnsets(st, monoOf(bd.samples, 0, C * 1920)), k * C * 4, P0 * 4, false);
                const probe = wait((done) => yue.inpaint(p.plan, F, new Float32Array(F.length * 64), 0, F.length, {
                    ...(s.abc ? { abc: s.abc } : {}), style: s.style, lyrics: s.lyrics, cot, seed: seed * 1000 + 900 + k,
                    odeSteps: ode, onDone: done }), st.name + ' probe ' + (k + 1));
                const pd = vae.decode(probe.latents.slice(0, F.length * 64));
                const fb = fitBeat(beatOnsets(st, monoOf(pd.samples, 0, F.length * 1920)), 0, P0 * 4, false);
                let dd = ((k + 1) * C + fb.phase * P0 - sb.phase * P0) % P0;
                if (dd < 0) dd += P0;
                const cut = Math.min(span, Math.round(dd));
                rows[k].spliceSeconds = tb - t1;
                rows[k + 1].spliceSeconds = t1 - t;
                rows[k].memory = before.length; rows[k].bridge = true; rows[k].scores = Array.from(r.scores);
                rows[k + 1].trim = cut;
                rows[k + 1].phaseFit = { P: P0, streamPhase: sb.phase, freshPhase: fb.phase, cut, rendered: (k + 1) * C,
                                         seconds: now() - tb };
                rows[k + 1].memory = 0; rows[k + 1].fresh = true; rows[k + 1].lyrics = rows[k].lyrics;
                if (roll) rows[k + 1].lyricLines = rows[k].lyricLines;
                append(bridgeTok);
                append(F.slice(cut, cut + C));
                // st.keepProbe: the probe render is the render (the last old chunk and
                // the bridge), not rendered again: a second NAR draw can put the
                // accents half a beat elsewhere (4 of exp7f's 5 misses).
                if (st.keepProbe) {
                    if (lat.length < (k + 1) * C * 64) { const nl = new Float32Array(tokens.length * 64); nl.set(lat); lat = nl; }
                    lat.set(pr.latents.subarray((r1 - s0) * 64, ((k + 1) * C - s0) * 64), r1 * 64);
                    const probeSeconds = tb === undefined ? 0 : (now() - tb);
                    for (let j = narDone; j <= k; j++) {
                        rows[j].inpaintSeconds = probeSeconds / (k + 1 - narDone);
                        rows[j].narWindow = win.length; rows[j].probeRender = true;
                        rows[j].latentNorm = latentNorm(lat, j * C, (j + 1) * C);
                        const h0 = Math.max(0, j * C - 12), h1 = Math.min((j + 1) * C + 12, (k + 1) * C);
                        const td = now();
                        const dj = vae.decode(lat.slice(h0 * 64, h1 * 64));
                        rows[j].decodeSeconds = now() - td;
                        if (st.beatSensor === 'audio') chunkMono[j] = monoOf(dj.samples, (j * C - h0) * 1920, C * 1920);
                    }
                    narDone = k + 1;
                }
                k++;
                continue;
            }
            for (let i = 0; i < (cuts ? cuts.length : nT); i++) {
                const dTrim = cuts ? cuts[i] : Math.round(i * span / nT);
                const B = F.slice(dTrim, dTrim + C);
                const r = wait((done) => yue.splice(p.plan, Object.assign({
                    ...(s.abc ? { abc: s.abc } : {}), style: s.style, lyrics: s.lyrics, cot, seed: seed * 1000 + k,
                    before: before.slice(), middleFrames: C, after: B, candidates: st.bridgeCandidates || 4,
                    scoreFrames: 125, continue: false, onDone: done }, cfg)), st.name + ' bridge ' + k + ' trim ' + dTrim);
                const sc = Math.min(...r.scores);
                trials.push({ trim: dTrim, scores: Array.from(r.scores) });
                if (!best || sc < best.sc) best = { sc, r, B, dTrim };
            }
            rows[k].spliceSeconds = now() - t1;
            rows[k + 1].spliceSeconds = t1 - t;
            rows[k].memory = before.length; rows[k].bridge = true; rows[k].scores = Array.from(best.r.scores);
            rows[k].trials = trials; rows[k + 1].trim = best.dTrim;
            rows[k + 1].memory = 0; rows[k + 1].fresh = true; rows[k + 1].lyrics = rows[k].lyrics;
            if (roll) rows[k + 1].lyricLines = rows[k].lyricLines;
            append(best.r.semantic.subarray(best.r.middle0, best.r.middle0 + C));
            append(best.B);
            k++;
            continue;
        }
        const t = now();
        const r = wait((done) => yue.splice(p.plan, Object.assign({
            ...(s.abc ? { abc: s.abc } : {}), style: s.style, lyrics: s.lyrics, cot, seed: seed * 1000 + k,
            before: before.slice(), middleFrames: C, candidates: 1, continue: false,
            onDone: done }, cfg)), st.name + ' splice ' + k);
        rows[k].spliceSeconds = now() - t;
        rows[k].memory = before.length;
        const m = r.semantic.subarray(r.middle0, r.middle0 + C);
        if (m.length !== C) throw new Error(st.name + ' chunk ' + k + ': ' + m.length + ' tokens, wanted ' + C);
        append(m);
    }
    pump(true);
    const T = tokens.length;

    const a = vae.decode(lat, { coreFrames: 500 });
    writeBin(dir + '/tokens.i32', tokens);
    writeBin(dir + '/latents.f32', lat);
    writeBin(dir + '/audio.f32', a.samples);
    writeJson(dir + '/chunks.json', {
        name: st.name, memory: M, chunk: C, lookahead: LA, narLeft: NL, cot, seed, odeSteps: ode,
        cfgScale: st.cfgScale, bridge: !!st.bridge, changeMemory: st.changeMemory, lyricsRoll: roll,
        frames: T, audioFrames: a.frames, wall: now() - t0,
        plans: Object.fromEntries([...plans].map(([k, v]) => [k, { seconds: v.seconds, prefix: v.plan.prefix.length }])),
        schedule: st.schedule, chunks: rows,
    });
    const avg = (f) => rows.reduce((x, r) => x + r[f], 0) / K;
    console.log(st.name + ': ' + K + ' chunks, ' + (now() - t0).toFixed(1) + ' s wall; per chunk splice ' +
                avg('spliceSeconds').toFixed(2) + ' inpaint ' + avg('inpaintSeconds').toFixed(2) + ' decode ' +
                avg('decodeSeconds').toFixed(3) + ' s; norm ' + rows.map((r) => r.latentNorm.toFixed(1)).join(' '));
}
console.log('done');

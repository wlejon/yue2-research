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

function wait(start, desc) {
    let result = null, info = null;
    const h = start((r, i) => { result = r; info = i || {}; });
    L.pumpUntil(() => info !== null, 3600, desc);
    if (info.error) throw new Error(desc + ': ' + info.error);
    if (info.cancelled || !result) throw new Error(desc + ': cancelled');
    return result;
}

const plans = new Map();
function planFor(key, cot) {
    const k = key + '|' + cot;
    if (plans.has(k)) return plans.get(k);
    const s = job.styles[key];
    const t = now();
    const r = wait((done) => yue.plan({ style: s.style, lyrics: s.lyrics, cot, seed: 1, onDone: done }), 'plan ' + key);
    const p = { plan: r.plan, seconds: now() - t };
    plans.set(k, p);
    return p;
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

    // Semantic: chunk after chunk, each opened by the last M tokens.
    let tokens = new Int32Array(0);
    for (let k = 0; k < K; k++) {
        const s = job.styles[order[k]];
        const p = planFor(order[k], cot);
        const before = tokens.subarray(Math.max(0, tokens.length - M));
        const t = now();
        const r = wait((done) => yue.splice(p.plan, {
            style: s.style, lyrics: s.lyrics, cot, seed: seed * 1000 + k,
            before: before.slice(), middleFrames: C, candidates: 1, continue: false,
            onDone: done }), st.name + ' splice ' + k);
        rows[k].spliceSeconds = now() - t;
        rows[k].memory = before.length;
        const m = r.semantic.subarray(r.middle0, r.middle0 + C);
        if (m.length !== C) throw new Error(st.name + ' chunk ' + k + ': ' + m.length + ' tokens, wanted ' + C);
        const nt = new Int32Array(tokens.length + C);
        nt.set(tokens); nt.set(m, tokens.length);
        tokens = nt;
    }

    // NAR: each chunk inpainted over its window, the lookahead solved and dropped.
    const T = tokens.length;
    const lat = new Float32Array(T * 64);
    for (let k = 0; k < K; k++) {
        const s = job.styles[order[k]];
        const p = planFor(order[k], cot);
        const c0 = k * C, c1 = c0 + C, e1 = Math.min(T, c1 + LA);
        const s0 = Math.max(0, c0 - NL);
        const win = tokens.slice(s0, e1);
        const wl = new Float32Array(win.length * 64);
        wl.set(lat.subarray(s0 * 64, c0 * 64));
        const t = now();
        const r = wait((done) => yue.inpaint(p.plan, win, wl, c0 - s0, e1 - s0, {
            style: s.style, lyrics: s.lyrics, cot, seed: seed * 1000 + 500 + k, odeSteps: ode,
            onDone: done }), st.name + ' inpaint ' + k);
        rows[k].inpaintSeconds = now() - t;
        rows[k].narWindow = win.length;
        lat.set(r.latents.subarray((c0 - s0) * 64, (c1 - s0) * 64), c0 * 64);
        rows[k].latentNorm = latentNorm(lat, c0, c1);
        // A chunk's decode, as a stream would run it (halo on both sides).
        const h0 = Math.max(0, c0 - 12), h1 = Math.min(c1 + 12, (k + 1) * C);
        const td = now();
        vae.decode(lat.slice(h0 * 64, h1 * 64));
        rows[k].decodeSeconds = now() - td;
    }

    const a = vae.decode(lat, { coreFrames: 500 });
    writeBin(dir + '/tokens.i32', tokens);
    writeBin(dir + '/latents.f32', lat);
    writeBin(dir + '/audio.f32', a.samples);
    writeJson(dir + '/chunks.json', {
        name: st.name, memory: M, chunk: C, lookahead: LA, narLeft: NL, cot, seed, odeSteps: ode,
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

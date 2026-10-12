// Exp 1a/3 latency components on the native path, from a finished run:
//   vae      sync decode time for tiles of n frames (and the whole song)
//   window   synthesize() of one NAR window of w frames (prompt prefill + codec
//            prefill + ODE), at several w and ODE step counts
//   prefill  semantic TTFT (prefill + first token) for prompts of growing length,
//            made long by repeating the plan's ABC (what a re-prompted
//            continuation pays to prefill prompt + the song's tokens so far)
// Job: { "from": "runs/exp1/folk_gen_W0", "out": "tables/latency_native.json" }
const L = require('./native/lib.js');
const { fs, now } = L;

const job = L.readJob();
const from = L.abs(job.from);
const yue = L.loadYue(job.arWeights || 'int8');
const vae = L.loadVae();

function bytes(path) {
    const b = fs.readFileSync(path);
    return b instanceof ArrayBuffer ? new Uint8Array(b) : new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
}
const lat = (() => { const u = bytes(from + '/latents.f32').slice(); return new Float32Array(u.buffer); })();
const sem = (() => { const u = bytes(from + '/semantic.i32').slice(); return new Int32Array(u.buffer); })();
const pj = JSON.parse(fs.readFileSync(from + '/plan.json', 'utf-8'));
const plan = { abc: pj.abc, abcIds: new Int32Array(pj.abcIds), prefix: new Int32Array(pj.prefix), truncated: false };
const base = { style: pj.style, lyrics: pj.lyrics, cot: pj.cot, abc: pj.abc, seed: pj.seed };
const out = { vae: [], window: [], prefill: [] };

function med(xs) { const s = xs.slice().sort((a, b) => a - b); return s[s.length >> 1]; }
function run(fn, desc) {
    let res = null, info = null;
    const t = now();
    fn((r, i) => { res = r; info = i; });
    L.pumpUntil(() => info !== null, 600, desc);
    if (info.error) throw new Error(desc + ': ' + info.error);
    return { res, secs: now() - t };
}

// VAE
for (const n of [8, 38, 50, 100, 250]) {
    const z = lat.subarray(0, (n + 24) * 64);    // core + 12-frame halo each side
    const ts = [];
    for (let i = 0; i < 5; i++) { const t = now(); vae.decode(z); ts.push(now() - t); }
    out.vae.push({ frames: n + 24, seconds: med(ts) });
    console.log('vae ' + (n + 24) + ' frames: ' + med(ts).toFixed(3) + ' s');
}
{
    const t = now(); vae.decode(lat, { coreFrames: 250 }); const s = now() - t;
    out.vae.push({ frames: lat.length / 64, seconds: s, whole: true });
    console.log('vae whole song ' + (lat.length / 64) + ' frames: ' + s.toFixed(2) + ' s');
}

// NAR windows (one window = synthesize of w tokens: prompt + codec prefill + ODE)
for (const steps of [32, 16, 8]) {
    for (const w of [25, 50, 100, 250]) {
        const ts = [];
        for (let i = 0; i < 3; i++) {
            const r = run((done) => yue.synthesize(plan, sem.slice(1000, 1000 + w),
                                                   Object.assign({}, base, { odeSteps: steps, onDone: done })), 'window');
            ts.push(r.secs);
        }
        out.window.push({ frames: w, steps, seconds: med(ts), prefix: plan.prefix.length });
        console.log('NAR window ' + w + ' frames, ' + steps + ' steps: ' + med(ts).toFixed(3) + ' s');
    }
}

// Semantic prefill vs prompt length (ABC repeated k times)
for (const k of [1, 2, 3, 4]) {
    const abc = Array(k).fill(pj.abc).join('\n');
    const o = Object.assign({}, base, { abc, semanticSampling: { minTokens: 0, maxTokens: 2 } });
    const pl = run((done) => yue.plan(Object.assign({}, o, { onDone: done })), 'plan').res.plan;
    const ts = [];
    for (let i = 0; i < 3; i++) {
        const r = run((done) => yue.semantic(pl, Object.assign({}, o, { onDone: done })), 'semantic');
        ts.push(r.secs);   // semantic() reports no timing: wall of prefill + 2 tokens (+ a few ms of event pumping)
    }
    out.prefill.push({ prefix: pl.prefix.length, ttft: med(ts) });
    console.log('semantic TTFT at prefix ' + pl.prefix.length + ': ' + med(ts).toFixed(3) + ' s');
}
L.writeJson(L.abs(job.out), out);
console.log('done');

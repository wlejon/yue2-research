// Native generation runs: one model load, a list of runs, every artifact and a
// callback timeline written per run.
//
// Job (YUE_JOB):
// { "out": "runs/xyz", "arWeights": "int8", "vae": true,
//   "runs": [ { "name": "a", "mode": "generate" | "synthesize",
//               "from": "runs/xyz/a"   (synthesize: plan.json + semantic.i32 of a run),
//               "slice": [a, b]        (synthesize: only semantic[a, b)),
//               "style", "lyrics", "cot", "seed", "abc", "cfgScale", "odeSteps", "narChunkFrames",
//               "semanticSampling": {...}, "abcSampling": {...},
//               "steer": [{ "layer", "nar", "scale", "delta": "path.f32", ... }],
//               "capture": { "layers": [...], "plan", "semantic", "narEvals": [...] },
//               "keepHidden": false, "decodeFirstFrames", "decodeCoreFrames", "saveAudio": true } ] }
// A headless script's own require resolves against the working directory
// (the repo root), not the script's directory: see FINDINGS "bugs found".
const L = require('./native/lib.js');
const { fs, now, writeBin, writeJson, mkdirp } = L;

const job = L.readJob();
const yue = L.loadYue(job.arWeights);
const vae = job.vae ? L.loadVae() : null;

// readFileSync may hand back a byte view rather than a bare ArrayBuffer, and
// new Int32Array(bytes) would convert each byte; view the bytes instead.
function bytes(path) {
    const b = fs.readFileSync(path);
    return b instanceof ArrayBuffer ? new Uint8Array(b) : new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
}
function readF32(path) {
    const u = bytes(path).slice();
    return new Float32Array(u.buffer, 0, u.byteLength >> 2);
}
function readI32(path) {
    const u = bytes(path).slice();
    return new Int32Array(u.buffer, 0, u.byteLength >> 2);
}

function opts(run, tl, t0) {
    const o = {};
    for (const k of ['style', 'lyrics', 'cot', 'seed', 'abc', 'cfgScale', 'odeSteps', 'narChunkFrames',
                     'semanticSampling', 'abcSampling', 'runSemantic', 'runSynthesize', 'reusePrefixKv',
                     'abcDraftTokens', 'decodeFirstFrames', 'decodeCoreFrames', 'decodeHaloFrames'])
        if (run[k] !== undefined) o[k] = run[k];
    if (run.steer) o.steer = run.steer.map((s) => Object.assign({}, s, { delta: readF32(s.delta) }));
    if (run.capture) o.capture = run.capture;
    if (vae && run.decode !== false) o.vae = vae;
    let nsem = 0, nlat = 0, naud = 0;
    o.onAbc = (text) => { if (!tl.abc0) tl.abc0 = now() - t0; };
    o.onSemantic = (ids) => {
        if (!nsem) tl.sem0 = now() - t0;
        nsem += ids.length;
        tl.sem.push([now() - t0, nsem]);
    };
    o.onLatents = (c) => {
        if (!nlat) tl.lat0 = now() - t0;
        nlat = c.frame0 + c.frames;
        tl.lat.push([now() - t0, c.frame0, c.frames]);
    };
    o.onAudio = (c) => {
        if (!naud) tl.aud0 = now() - t0;
        naud = c.sample0 + c.frames;
        tl.aud.push([now() - t0, c.sample0, c.frames]);
    };
    return o;
}

function saveHidden(dir, hidden, keep) {
    // Mean per (stage, layer, narEval); optionally every row.
    const groups = new Map();
    for (const h of hidden) {
        const key = h.stage + ':' + h.layer + ':' + h.narEval;
        let g = groups.get(key);
        if (!g) { g = { stage: h.stage, layer: h.layer, narEval: h.narEval, rows: 0, sum: new Float64Array(h.hidden), blocks: [] }; groups.set(key, g); }
        for (let r = 0; r < h.rows; r++)
            for (let i = 0; i < h.hidden; i++) g.sum[i] += h.data[r * h.hidden + i];
        g.rows += h.rows;
        if (keep) g.blocks.push({ pos0: h.pos0, rows: h.rows, data: h.data });
    }
    const keys = [];
    const means = [];
    for (const g of groups.values()) {
        keys.push({ stage: g.stage, layer: g.layer, narEval: g.narEval, rows: g.rows });
        const m = new Float32Array(g.sum.length);
        for (let i = 0; i < m.length; i++) m[i] = g.sum[i] / g.rows;
        means.push(m);
        if (keep) {
            g.blocks.sort((a, b) => a.pos0 - b.pos0);
            writeBin(dir + '/hidden_' + g.stage + '_L' + g.layer + '_e' + g.narEval + '.f32',
                     L.concat(g.blocks.map((b) => b.data), Float32Array));
        }
    }
    writeJson(dir + '/hidden_keys.json', keys);
    writeBin(dir + '/hidden_means.f32', L.concat(means, Float32Array));
}

for (const run of job.runs) {
    const dir = job.out + '/' + run.name;
    if (fs.existsSync(dir + '/run.json') && !job.force) { console.log(run.name + ': exists, skipped'); continue; }
    mkdirp(dir);
    const tl = { sem: [], lat: [], aud: [] };
    const audio = [];
    let result = null, info = null;
    const t0 = now();
    const o = opts(run, tl, t0);
    const keepAudio = run.saveAudio !== false && o.vae;
    if (keepAudio) {
        const inner = o.onAudio;
        o.onAudio = (c) => { inner(c); audio.push(c.samples.slice()); audio.push(c.frames); };
    }
    o.onDone = (r, i) => { result = r; info = i; };
    let h;
    if ((run.mode || 'generate') === 'generate') {
        h = yue.generate(o);
    } else {
        const pj = JSON.parse(fs.readFileSync(run.from + '/plan.json', 'utf-8'));
        const plan = { abc: pj.abc, abcIds: new Int32Array(pj.abcIds), prefix: new Int32Array(pj.prefix), truncated: pj.truncated };
        let sem = readI32(run.from + '/semantic.i32');
        if (run.slice) sem = sem.slice(run.slice[0], run.slice[1]);
        if (o.style === undefined) o.style = pj.style;
        if (o.lyrics === undefined) o.lyrics = pj.lyrics;
        if (o.cot === undefined) o.cot = pj.cot;
        if (o.abc === undefined && pj.abc) o.abc = pj.abc;
        if (o.seed === undefined) o.seed = pj.seed;
        h = yue.synthesize(plan, sem, o);
    }
    L.pumpUntil(() => result !== null || info !== null, 3600, run.name);
    const wall = now() - t0;
    if (info && info.error) throw new Error(run.name + ': ' + info.error);
    const r = result;
    writeJson(dir + '/plan.json', {
        abc: r.plan ? r.plan.abc : '', abcIds: r.plan ? Array.from(r.plan.abcIds) : [],
        prefix: r.plan ? Array.from(r.plan.prefix) : [], truncated: r.plan ? r.plan.truncated : false,
        style: o.style, lyrics: o.lyrics, cot: o.cot || 'full', seed: o.seed === undefined ? 831001 : o.seed,
    });
    if (r.semantic && r.semantic.length) writeBin(dir + '/semantic.i32', r.semantic);
    if (r.latents && r.latents.length) writeBin(dir + '/latents.f32', r.latents);
    if (keepAudio && audio.length) {
        // Planar chunks -> two planar channels in one file: [L..., R...].
        let n = 0;
        for (let i = 1; i < audio.length; i += 2) n += audio[i];
        const out = new Float32Array(2 * n);
        let o2 = 0;
        for (let i = 0; i < audio.length; i += 2) {
            const s = audio[i], f = audio[i + 1];
            out.set(s.subarray(0, f), o2);
            out.set(s.subarray(f, 2 * f), n + o2);
            o2 += f;
        }
        writeBin(dir + '/audio.f32', out);
    }
    if (r.hidden && r.hidden.length) saveHidden(dir, r.hidden, run.keepHidden);
    writeJson(dir + '/run.json', { name: run.name, wall, cancelled: r.cancelled, frames: r.frames,
                                   timing: r.timing, timeline: tl, semanticTruncated: r.semanticTruncated,
                                   options: Object.assign({}, run, { lyrics: undefined }) });
    console.log(run.name + ': ' + wall.toFixed(2) + ' s wall, frames ' + r.frames + ', sem0 ' +
                (tl.sem0 || 0).toFixed(2) + ' lat0 ' + (tl.lat0 || 0).toFixed(2) + ' aud0 ' + (tl.aud0 || 0).toFixed(2) +
                ' timing ' + JSON.stringify(r.timing));
}
console.log('done');

// A paced run of bro.music's endless stream (yue.endless), played through
// bro's audio engine: the program audio goes into one engine stream
// (createStream / pushStreamSamples) and the playhead the stream is given is
// what the mixer has consumed of it (getStreamStats().playedFrames), the
// mixer pumped by advanceTime() at the wall clock's rate. A script of
// controls runs on the program clock; every event, the latency of each
// control (when the stream said it would be heard, when it was) and the
// stream's service gaps go to <out>/paced.json, the program audio to
// <out>/program.f32 (planar L then R).
//
//   YUE_PACED=branches|score YUE_OUT=runs/endless_check/paced_branches \
//       bro-headless native native/endless_paced.js
//
// The stream is instrumental (no lyrics: exp9's rolled-lyrics paced runs were
// brosoundml e3e81eb's 'lyrics' mode).
// branches: the exp7b voice style, changed to synth, knobs, a branch to metal
//   committed, a branch to lofi discarded, back to folk; 4 minutes.
// score: the exp7h folk score as synth, then metal, jazz and orchestral, each
//   change a measured bridge; 3 minutes.
const L = require('./lib.js');
const { fs, now, writeBin, writeJson, mkdirp } = L;

const mode = process.env.YUE_PACED || 'branches';
if (mode !== 'branches' && mode !== 'score') throw new Error('YUE_PACED must be branches or score');
const out = L.abs(process.env.YUE_OUT || 'runs/endless_check/paced_' + mode);
mkdirp(out);
const jobs = { b: JSON.parse(fs.readFileSync(L.abs('jobs/exp7b.json'), 'utf8')),
               h: JSON.parse(fs.readFileSync(L.abs('jobs/exp7h.json'), 'utf8')) };
const yue = L.loadYue('int8');
const vae = L.loadVae();
const SR = 48000;

let st = null;
const log = [], controls = [];
function control(at, what, fn) { controls.push({ at, what, fn, done: false }); }

let opts, total;
if (mode === 'branches') {
    const sty = (k) => jobs.b.styles[k].style;
    opts = { memory: 500, target: { style: sty('voice'), tag: 'voice' } };
    let br1 = -1, br2 = -1;
    control(40, 'setTarget synth', () => st.setTarget({ style: sty('synth'), tag: 'synth' }));
    control(70, 'setKnobs energy 0.6 texture 1', () => st.setKnobs({ energy: 0.6, texture: 1 }));
    control(95, 'branch metal', () => { br1 = st.branch({ style: sty('metal'), tag: 'metal' }); return br1; });
    control(125, 'commitBranch metal', () => st.commitBranch(br1));
    control(150, 'branch lofi', () => { br2 = st.branch({ style: sty('lofi'), tag: 'lofi' }); return br2; });
    control(175, 'discardBranch lofi', () => st.discardBranch(br2));
    control(185, 'setTarget folk + knobs 0', () => st.setTarget({ style: sty('folk'), tag: 'folk', knobs: { energy: 0, texture: 0 } }));
    total = 240;
} else {
    const sty = jobs.h.styles;
    const t = (k) => ({ style: sty[k].style, score: sty[k].abc, tag: k });
    opts = { memory: 250, target: t('score_synth') };
    control(30, 'setTarget score_metal', () => st.setTarget(t('score_metal')));
    control(70, 'setTarget score_jazz', () => st.setTarget(t('score_jazz')));
    control(110, 'setTarget score_orch', () => st.setTarget(t('score_orch')));
    total = 180;
}

// ── the player ──
const ctx = new AudioContext();
const ER = ctx.sampleRate || SR;
let pb = -1, pushed = 0, contiguous = true, queue = [];
// A streaming linear resampler, program rate to engine rate: output frame j
// sits at program sample j * SR / ER.
let rsOut = 0, rsPrevL = 0, rsPrevR = 0;
function resampleChunk(l, r, sample0) {
    const end = sample0 + l.length;
    const n = Math.max(0, Math.floor(((end - 1) * ER) / SR) + 1 - rsOut);
    const x = new Float32Array(2 * n);
    for (let j = 0; j < n; j++) {
        const pos = ((rsOut + j) * SR) / ER - sample0;
        const i = Math.floor(pos), f = pos - i;
        const l0 = i < 0 ? rsPrevL : l[i], r0 = i < 0 ? rsPrevR : r[i];
        const l1 = i + 1 < l.length ? l[i + 1] : l0, r1 = i + 1 < l.length ? r[i + 1] : r0;
        x[2 * j] = l0 + (l1 - l0) * f;
        x[2 * j + 1] = r0 + (r1 - r0) * f;
    }
    rsOut += n;
    rsPrevL = l[l.length - 1]; rsPrevR = r[r.length - 1];
    return x;
}
const progL = [], progR = [];
function drain() {
    while (queue.length) {
        const head = queue[0];
        const wrote = ctx.pushStreamSamples(pb, head);
        if (wrote * 2 >= head.length) { queue.shift(); continue; }
        if (wrote > 0) queue[0] = head.subarray(wrote * 2);
        break;
    }
}
function played() {
    if (pb < 0) return 0;
    return Math.min(pushed, Math.round((ctx.getStreamStats(pb).playedFrames * SR) / ER));
}

let done = null, t0 = now(), tFirst = -1;
opts.vae = vae;
if (process.env.YUE_AHEAD) opts.aheadSeconds = +process.env.YUE_AHEAD;   // the least sample lead (s)
opts.onAudio = (a) => {
    if (a.sample0 !== pushed) contiguous = false;
    const l = a.samples.slice(0, a.frames), r = a.samples.slice(a.frames, 2 * a.frames);
    progL.push(l); progR.push(r);
    queue.push(resampleChunk(l, r, a.sample0));
    pushed = a.sample0 + a.frames;
    if (pb < 0) { pb = ctx.createStream(2, 4 * ER); tFirst = now() - t0; }
    drain();
};
opts.onEvent = (e) => {
    const r = { kind: e.kind, chunk: e.chunk, branch: e.branch, tag: e.tag, sample: e.sample, value: e.value,
                text: e.text, wall: now() - t0, playhead: played() / SR };
    if (e.row && e.kind === 'built')
        Object.assign(r, { how: e.row.how, memory: e.row.memory, retargeted: e.row.retargeted,
                           spliceSeconds: e.row.spliceSeconds, inpaintSeconds: e.row.inpaintSeconds });
    if (e.row && e.row.phaseFit) r.phaseFit = e.row.phaseFit;
    log.push(r);
    if (e.kind !== 'chunkStart' && e.kind !== 'sampled' && e.kind !== 'built')
        console.log(r.playhead.toFixed(1).padStart(6) + ' s  ' + e.kind + ' chunk ' + e.chunk + ' ' + (e.tag || '') +
                    (r.text ? ' "' + String(r.text).slice(0, 80).replace(/\n/g, ' / ') + '"' : '') +
                    (e.kind === 'planned' ? ' heard at ' + (e.sample / SR).toFixed(1) + ' s' : ''));
};
opts.onDone = (i) => { done = i; };
st = yue.endless(opts);

// ── the clock: the mixer pumped at the wall clock's rate ──
let last = now(), maxTick = 0, underrunAtFirst = -1, stopped = false, overran = false;
const stallTicks = [];
while (done === null) {
    wallSleep(4);
    const t = now();
    const dt = (t - last) * 1000;
    last = t;
    if (dt > 100) stallTicks.push({ wall: t - t0, ms: dt });
    maxTick = Math.max(maxTick, dt);
    advanceTime(dt);
    if (pb >= 0) {
        drain();
        st.setPlayhead(played());
        if (underrunAtFirst < 0) underrunAtFirst = ctx.getStreamStats(pb).underrunFrames;
    }
    const ph = played() / SR;
    for (const c of controls)
        if (!c.done && ph >= c.at) {
            c.done = true;
            c.wall = now() - t0;
            c.playhead = ph;
            const res = c.fn();
            if (typeof res === 'number' || typeof res === 'boolean') c.result = res;
            console.log(ph.toFixed(1).padStart(6) + ' s  >> ' + c.what);
        }
    if (ph >= total && !stopped) { st.stop(); stopped = true; }
    if (!overran && now() - t0 > total + 120) {
        overran = true;
        console.log('OVERRAN: ' + JSON.stringify(Object.assign({}, st.stats, { chunks: undefined, target: undefined })));
        st.stop();
    }
}
const ss = pb >= 0 ? ctx.getStreamStats(pb) : null;
const stats = st.stats;

// Each control's latency: the 'planned' (or branch/committed) event it caused and the
// first chunk with its tag that started playing.
for (const c of controls) {
    const after = log.filter((e) => e.wall >= c.wall);
    const pl = after.find((e) => e.kind === 'planned' || e.kind === 'knobs' || e.kind === 'branch' || e.kind === 'committed' ||
                                 e.kind === 'discarded');
    c.event = pl ? { kind: pl.kind, chunk: pl.chunk, at: pl.sample / SR, wall: pl.wall - c.wall } : null;
    if (pl && pl.kind === 'planned') {
        const cs = after.find((e) => e.kind === 'chunkStart' && e.chunk === pl.chunk && e.branch === 0);
        c.heard = cs ? { at: cs.sample / SR, after: cs.sample / SR - c.playhead } : null;
    }
    if (pl && pl.kind === 'committed') c.heard = { at: pl.sample / SR, after: pl.sample / SR - c.playhead };
    delete c.fn;
}
writeBin(out + '/program.f32', L.concat(progL.concat(progR), Float32Array));
const summary = {
    mode, total, overran, wall: now() - t0, firstAudioSeconds: tFirst, programSeconds: pushed / SR, contiguous,
    underrunSeconds: ss ? (ss.underrunFrames - Math.max(0, underrunAtFirst)) / ER : null, engineRate: ER,
    worstGapSeconds: stats.worstGapSeconds, gaps: stats.gaps, work: stats.work, cost: stats.costSeconds,
    maxJsTickMs: maxTick, stallTicks, controls, done,
    chunks: stats.chunks.map((c) => ({ index: c.index, tag: c.tag, how: c.how, memory: c.memory, retargeted: c.retargeted })),
};
writeJson(out + '/paced.json', Object.assign({}, summary, { events: log }));
console.log(JSON.stringify(Object.assign({}, summary, { chunks: undefined, stallTicks: stallTicks.length }), null, 1));

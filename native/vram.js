// VRAM breakdown, one scenario per process (YUE_JOB = { scenario, out, ... }).
// Device-wide used bytes on cuda:0 (bro.gpu.memoryInfo = cudaMemGetInfo, so it
// includes the CUDA context, the Vulkan device and brotensor's cached-but-free
// blocks) sampled after each load and at every callback of a generate; the
// peak per stage is the most used seen while that stage's callbacks fire.
const L = require('./native/lib.js');
const job = L.readJob();
const dev = 'cuda:0';
const used = () => { const m = bro.gpu.memoryInfo(dev); return (m.totalBytes - m.freeBytes) / 2 ** 30; };
const rec = { scenario: job.scenario, marks: {}, peaks: {} };
const mark = (k) => { rec.marks[k] = used(); console.log(k + ': ' + rec.marks[k].toFixed(3) + ' GiB'); };
const peak = (k) => { const u = used(); if (!(rec.peaks[k] >= u)) rec.peaks[k] = u; };

bro.music.init();
mark('start');
const s = job.song || {};
let vae = null, yue = null;
if (job.vae) { vae = L.loadVae(); mark('vae'); }
if (job.yue) { yue = L.loadYue(job.yue); mark('yue_' + job.yue); }
if (job.generate) {
    let info = null;
    const o = Object.assign({ style: s.style, lyrics: s.lyrics, cot: 'full', seed: 1001 }, job.generate);
    if (vae) o.vae = vae;
    let n = 0;
    o.onAbc = () => peak('plan');
    o.onSemantic = (ids) => { n += ids.length; peak('semantic'); };
    o.onLatents = () => peak('synthesize');
    o.onAudio = () => peak('decode');
    o.onDone = (r, i) => { info = i; rec.frames = r.frames; rec.semanticTokens = r.timing.semanticTokens; };
    yue.generate(o);
    L.pumpUntil(() => {
        if (info === null) peak('running');
        return info !== null;
    }, 1800, 'generate');
    mark('after');
}
L.writeJson(L.abs(job.out), rec);
console.log(JSON.stringify(rec));

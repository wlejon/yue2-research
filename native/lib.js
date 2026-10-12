// Shared helpers for the native (bro-headless + bro.music) harness.
//
// Every script reads its job from a JSON file named by YUE_JOB (paths in it
// are relative to the repo root, the working directory) and writes binary
// artifacts the Python side reads with numpy.fromfile:
//   *.i32  int32 little-endian      *.f32  float32 little-endian
const fs = require('fs');

const MODEL_DIR = process.env.YUE2_MODEL || 'D:/projects/YuE2-3B';
const VAE_DIR = process.env.YUE2_VAE || 'D:/projects/brosoundml/weights/yue2-vae';

function now() { return perf.now() / 1000; }

// fs resolves a relative path against the app directory (native/), require
// against the working directory; every path a job names is made absolute
// against the working directory (the repo root) so both agree.
const ROOT = process.cwd().replace(/\\/g, '/');
function abs(p) { return /^([A-Za-z]:)?[\\/]/.test(p) ? p : ROOT + '/' + p; }

function readJob() {
    const p = process.env.YUE_JOB;
    if (!p) throw new Error('YUE_JOB not set');
    const job = JSON.parse(fs.readFileSync(abs(p), 'utf-8'));
    job.out = abs(job.out);
    for (const r of job.runs || []) {
        if (r.from) r.from = abs(r.from);
        for (const s of r.steer || []) if (typeof s.delta === 'string') s.delta = abs(s.delta);
    }
    return job;
}

function mkdirp(d) { fs.mkdirSync(d, { recursive: true }); }

function writeBin(path, typed) {
    fs.writeFileSync(path, new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength));
}

function writeJson(path, obj) { fs.writeFileSync(path, JSON.stringify(obj, null, 1)); }

// Pump the event loop until fn() is true. Callbacks from async jobs are
// delivered on loop ticks, so a short sleep keeps their timestamps tight.
function pumpUntil(fn, seconds, desc) {
    const end = now() + seconds;
    while (!fn()) {
        if (now() > end) throw new Error('timeout waiting for ' + (desc || 'job'));
        advanceTime(4);
        wallSleep(2);
    }
}

let _yue = null, _vae = null;
function loadYue(arWeights) {
    if (!_yue) {
        const t = now();
        _yue = bro.music.loadYuE2(MODEL_DIR, { arWeights: arWeights || 'int8' });
        console.log('loaded YuE2 (' + _yue.arWeights + ') in ' + (now() - t).toFixed(1) + ' s');
    }
    return _yue;
}
function loadVae() {
    if (!_vae) {
        const t = now();
        _vae = bro.music.loadVae(VAE_DIR, { decoderOnly: false });
        console.log('loaded VAE in ' + (now() - t).toFixed(1) + ' s');
    }
    return _vae;
}

// Concatenate a list of typed-array pieces.
function concat(pieces, Ctor) {
    let n = 0;
    for (const p of pieces) n += p.length;
    const out = new Ctor(n);
    let o = 0;
    for (const p of pieces) { out.set(p, o); o += p.length; }
    return out;
}

module.exports = { fs, abs, ROOT, now, readJob, mkdirp, writeBin, writeJson, pumpUntil, loadYue, loadVae, concat, MODEL_DIR, VAE_DIR };

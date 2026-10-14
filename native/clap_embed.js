// CLAP audio embeddings of every 10 s window of a run's audio, for style
// distances (exp5 mobility, exp6 style spread). For each run dir under the
// job's roots that has audio.f32 (planar L then R, 48 kHz) and no clap.f32:
// clap.f32 = windows x 512, unit length; window w covers [10 w, 10 w + 10) s.
// Also writes <root>/clap_text.json: the embedding of every job.texts prompt.
//
// Job (YUE_JOB): { "roots": ["runs/exp5"], "texts": ["indie folk ...", ...], "force": false }
const L = require('./lib.js');
const { fs, writeBin, writeJson } = L;

const job = L.readJob();
const clap = bro.ear.loadClap();
console.log('CLAP on ' + clap.device);
const W = 10 * 48000;

function readF32(path) {
    const b = fs.readFileSync(path);
    const u = (b instanceof ArrayBuffer ? new Uint8Array(b) : new Uint8Array(b.buffer, b.byteOffset, b.byteLength)).slice();
    return new Float32Array(u.buffer, 0, u.byteLength >> 2);
}

for (const r of job.roots) {
    const root = L.abs(r);
    if (job.texts && job.texts.length) {
        const e = clap.embedText(job.texts);
        writeJson(root + '/clap_text.json', job.texts.map((t, i) => ({ text: t, embedding: Array.from(e[i]) })));
    }
    const dirs = [];
    const walk = (d) => {
        for (const n of fs.readdirSync(d)) {
            const p = d + '/' + n;
            if (fs.statSync(p).isDirectory()) walk(p);
            else if (n === 'audio.f32') dirs.push(d);
        }
    };
    walk(root);
    for (const d of dirs) {
        if (fs.existsSync(d + '/clap.f32') && !job.force) continue;
        const a = readF32(d + '/audio.f32');
        const n = a.length >> 1;
        const m = new Float32Array(n);
        for (let i = 0; i < n; i++) m[i] = 0.5 * (a[i] + a[n + i]);
        // The decode ends a few samples short of whole chunks (1919936 for
        // 1000 frames), so a last window within 0.1 s of whole ends at the end.
        const K = Math.floor((n + 4800) / W);
        const out = new Float32Array(K * 512);
        for (let k = 0; k < K; k++) {
            const a0 = Math.min(k * W, n - W);
            out.set(clap.embedAudio({ samples: m.subarray(a0, a0 + W), sampleRate: 48000 }), k * 512);
        }
        writeBin(d + '/clap.f32', out);
        console.log(d.slice(root.length + 1) + ': ' + K + ' windows');
    }
}
console.log('done');

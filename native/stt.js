// Whisper large-v3 transcripts of every 10 s window, for whether a stream
// keeps singing its lyrics (exp5) and whether a separated vocal stays
// intelligible (exp7). For each run dir under the roots: stt.json =
// { mix: [text per window], vocals: [...] }, from audio.f32 (48 kHz planar
// stereo, averaged and decimated to 16 kHz) and stems/vocals_16k.f32 (sep.py).
//
// Job (YUE_JOB): { "roots": ["runs/exp5"], "match": "voice_", "force": false }
const L = require('./lib.js');
const { fs, writeJson } = L;

const job = L.readJob();
const dir = process.env.WHISPER || 'D:/projects/brosoundml/weights/whisper-large-v3';
const whisper = bro.stt.loadWhisper(dir);
const tok = bro.stt.loadTokenizer({ vocabPath: dir + '/vocab.json', mergesPath: dir + '/merges.txt',
                                    addedTokensPath: dir + '/added_tokens.json' });
const prompt = tok.buildPrompt('en', 'transcribe', false);

function readF32(path) {
    const b = fs.readFileSync(path);
    const u = (b instanceof ArrayBuffer ? new Uint8Array(b) : new Uint8Array(b.buffer, b.byteOffset, b.byteLength)).slice();
    return new Float32Array(u.buffer, 0, u.byteLength >> 2);
}

// 48 kHz planar stereo -> 16 kHz mono (a 3-tap box average before the decimation).
function mix16k(a) {
    const n = a.length >> 1, m = Math.floor(n / 3);
    const out = new Float32Array(m);
    for (let i = 0; i < m; i++) {
        let s = 0;
        for (let j = 0; j < 3; j++) s += a[3 * i + j] + a[n + 3 * i + j];
        out[i] = s / 6;
    }
    return out;
}

function windows(x) {
    const W = 160000, out = [];
    for (let k = 0; (k + 1) * W <= x.length; k++)
        out.push(tok.decode(whisper.transcribe({ samples: x.subarray(k * W, (k + 1) * W), sampleRate: 16000 }, prompt,
                                               { maxNewTokens: 120 }), true).trim());
    return out;
}

for (const r of job.roots) {
    const root = L.abs(r);
    for (const n of fs.readdirSync(root)) {
        const d = root + '/' + n;
        if (job.match && n.indexOf(job.match) < 0) continue;
        if (!fs.existsSync(d + '/audio.f32')) continue;
        if (fs.existsSync(d + '/stt.json') && !job.force) continue;
        const res = { mix: windows(mix16k(readF32(d + '/audio.f32'))) };
        if (fs.existsSync(d + '/stems/vocals_16k.f32')) res.vocals = windows(readF32(d + '/stems/vocals_16k.f32'));
        writeJson(d + '/stt.json', res);
        console.log(n + ': ' + res.mix.length + ' windows; ' + res.mix.slice(0, 3).join(' | '));
    }
}
console.log('done');

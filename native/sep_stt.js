// Exp 7 oracle 3 (sep_whisper.py drives it): the native separator against the
// oracle's vocal stem, heard by Whisper large-v3.
//
// Job (YUE_JOB):
//   { "phase": "separate", "run": "runs/exp7/roll_adapt_w4", "out": "<dir>" }
//     bro.music.loadSeparator (htdemucs_ft, vocals only) over the run's whole
//     audio.f32 (48 kHz planar stereo) -> <out>/vocals_native.f32 (planar)
//   { "phase": "stt", "out": "<dir>", "inputs": { "<name>": "<16 kHz mono f32>" } }
//     Whisper on every 10 s window of each input -> <out>/whisper.json
//     { <name>: [text per window] }
const L = require('./lib.js');
const { fs, writeJson } = L;

const job = L.readJob();
L.mkdirp(job.out);

function readF32(path) {
    const b = fs.readFileSync(path);
    const u = (b instanceof ArrayBuffer ? new Uint8Array(b) : new Uint8Array(b.buffer, b.byteOffset, b.byteLength)).slice();
    return new Float32Array(u.buffer, 0, u.byteLength >> 2);
}

if (job.phase === 'separate') {
    const dir = process.env.HTDEMUCS || 'D:/projects/brosoundml/weights/htdemucs_ft';
    const sep = bro.music.loadSeparator(dir, { stems: ['vocals'] });
    const a = readF32(L.abs(job.run) + '/audio.f32');
    let r = null, info = null;
    const t0 = L.now();
    sep.separate({ samples: a, channels: 2, sampleRate: 48000 }, {
        stems: 'vocals',
        onDone: (res, i) => { r = res; info = i; },
    });
    L.pumpUntil(() => info !== null, 600, 'separate');
    if (info.error || info.cancelled) throw new Error('separate failed: ' + JSON.stringify(info));
    const v = r.stems.vocals;
    L.writeBin(job.out + '/vocals_native.f32', v.samples);
    console.log('separated ' + (v.frames / 48000).toFixed(1) + ' s in ' + (L.now() - t0).toFixed(2) + ' s');
} else if (job.phase === 'stt') {
    const dir = process.env.WHISPER || 'D:/projects/brosoundml/weights/whisper-large-v3';
    const whisper = bro.stt.loadWhisper(dir);
    const tok = bro.stt.loadTokenizer({ vocabPath: dir + '/vocab.json', mergesPath: dir + '/merges.txt',
                                        addedTokensPath: dir + '/added_tokens.json' });
    const prompt = tok.buildPrompt('en', 'transcribe', false);
    const W = 160000, res = {};
    for (const [name, path] of Object.entries(job.inputs)) {
        const x = readF32(L.abs(path));
        res[name] = [];
        for (let k = 0; (k + 1) * W <= x.length; k++)
            res[name].push(tok.decode(whisper.transcribe({ samples: x.subarray(k * W, (k + 1) * W), sampleRate: 16000 },
                                                         prompt, { maxNewTokens: 120 }), true).trim());
        console.log(name + ': ' + res[name].length + ' windows');
    }
    writeJson(job.out + '/whisper.json', res);
} else {
    throw new Error('job.phase must be separate or stt');
}
console.log('done');

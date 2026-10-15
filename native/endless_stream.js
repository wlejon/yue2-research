// The check of brosoundml's endless stream (bro.music yue.endless) against
// endless.js: the same job file, the same chunk schedule and seeds, run
// through the stream unpaced, writing the same run-directory files
// (tokens.i32, latents.f32, audio.f32, chunks.json) plus program.f32 (the
// stream's own program audio, planar L then R, as it handed it out).
//
// A job stream maps onto the stream's options:
//   memory / chunk / lookahead / narLeft / seed / odeSteps / cfgScale   the same
//   schedule [[key, n], ...]       the first key's style is the target, each
//                                  later key a path entry at its first chunk
//   changeMemory                   changeMode 'drop' at that memory
//   bridge + phaseFrom 'bridge' + keepProbe + beatSensor 'spectral'
//                                  changeMode 'auto' under a score (the stream's bridge)
//   lyricsRoll { lines | sheet, window, adaptive }
//                                  the sheet rolled by the model's own lyric
//                                  pointer (attention head 14.10, FINDINGS 11);
//                                  each chunk's reading (words, pointer, mass,
//                                  lines reached) goes into its chunks.json row
// Anything else (fixed-rate rolls, energy or latent beat sensors, trims by
// NLL) is not the stream's and is refused.
//
//   YUE_JOB=jobs/endless_check.json YUE_OUT=runs/endless_check/stream \
//       bro-headless native native/endless_stream.js
const L = require('./lib.js');
const { fs, now, writeBin, writeJson, mkdirp } = L;

const job = L.readJob();
const outRoot = L.abs(process.env.YUE_OUT || job.out + '_stream');
const only = process.env.YUE_ONLY ? new Set(process.env.YUE_ONLY.split(',')) : null;
const yue = L.loadYue(job.arWeights);
const vae = L.loadVae();


function targetOf(st, key) {
    const s = job.styles[key];
    const t = { style: s.style, score: s.abc || null, lyrics: s.lyrics, roll: false, tag: key };
    if (st.lyricsRoll) {
        if (!st.lyricsRoll.adaptive) throw new Error(st.name + ': only adaptive lyric rolls are the stream\'s');
        t.lyrics = (st.lyricsRoll.sheet ? job.sheets[st.lyricsRoll.sheet] : st.lyricsRoll.lines).join('\n');
        t.roll = true;
        t.lyricWindow = st.lyricsRoll.window;
    }
    return t;
}

function optionsOf(st) {
    const C = st.chunk || 250;
    const o = {
        vae, paced: false, chunkData: true, retarget: false,
        chunkFrames: C, memory: st.memory, lookahead: st.lookahead === undefined ? 25 : st.lookahead,
        narLeft: st.narLeft === undefined ? 475 : st.narLeft, seed: st.seed || 1, odeSteps: st.odeSteps || 32,
        changeMode: 'plain', changeMemory: 0,
    };
    if (st.cfgScale !== undefined) o.cfgScale = st.cfgScale;
    if (st.changeMemory !== undefined) { o.changeMode = 'drop'; o.changeMemory = st.changeMemory; }
    if (st.bridge) {
        if (st.phaseFrom !== 'bridge' || !st.keepProbe || st.beatSensor !== 'spectral' || (st.trimSpan || 17) !== 17)
            throw new Error(st.name + ': only the measured bridge (phaseFrom bridge, keepProbe, spectral) is the stream\'s');
        o.changeMode = 'auto';
    }
    if ((st.cot || 'off') !== (job.styles[st.schedule[0][0]].abc ? 'full' : 'off'))
        throw new Error(st.name + ': cot must be full under a score and off without one');
    let k = 0;
    o.path = [];
    for (const [key, n] of st.schedule) {
        if (k === 0) o.target = targetOf(st, key);
        else o.path.push(Object.assign(targetOf(st, key), { at: k * C / 25 }));
        k += n;
    }
    o.maxChunks = k;
    return o;
}

for (const st of job.streams) {
    if (only && !only.has(st.name)) continue;
    const dir = outRoot + '/' + st.name;
    if (fs.existsSync(dir + '/chunks.json') && !job.force) { console.log(st.name + ': exists, skipped'); continue; }
    mkdirp(dir);
    const opts = optionsOf(st);
    const C = opts.chunkFrames, K = opts.maxChunks;
    const tokens = new Int32Array(K * C), lat = new Float32Array(K * C * 64);
    const rows = [], events = [];
    const progL = [], progR = [];
    let emitted = 0, contiguous = true, info = null, errors = [];
    const t0 = now();
    opts.onAudio = (a) => {
        if (a.sample0 !== emitted) contiguous = false;
        progL.push(a.samples.slice(0, a.frames));
        progR.push(a.samples.slice(a.frames, 2 * a.frames));
        emitted = a.sample0 + a.frames;
    };
    opts.onEvent = (e) => {
        if (e.kind === 'sampled') tokens.set(e.tokens, e.chunk * C);
        if (e.kind === 'built') { lat.set(e.latents, e.frame0 * 64); rows[e.chunk] = e.row; }
        if (e.kind === 'lyrics' || e.kind === 'phaseFit') rows[e.chunk] = Object.assign(rows[e.chunk] || {}, e.row);
        if (e.kind === 'error') errors.push(e.text);
        if (e.kind !== 'sampled' && e.kind !== 'built' && e.kind !== 'chunkStart')
            events.push({ kind: e.kind, chunk: e.chunk, text: e.text, value: e.value, t: now() - t0 });
    };
    opts.onDone = (i) => { info = i; };
    const s = yue.endless(opts);
    L.pumpUntil(() => info !== null, 7200, st.name + ' stream');
    if (errors.length) throw new Error(st.name + ': ' + errors.join('; '));
    const wall = now() - t0;
    const a = vae.decode(lat, { coreFrames: 500 });
    writeBin(dir + '/tokens.i32', tokens);
    writeBin(dir + '/latents.f32', lat);
    writeBin(dir + '/audio.f32', a.samples);
    writeBin(dir + '/program.f32', L.concat(progL.concat(progR), Float32Array));
    // endless.js's chunk rows, from the stream's.
    const order = [];
    for (const [key, n] of st.schedule) for (let i = 0; i < n; i++) order.push(key);
    const chunks = rows.map((r, k) => {
        const o = { k, style: order[k], f0: k * C, f1: (k + 1) * C, spliceSeconds: r.spliceSeconds,
                    inpaintSeconds: r.inpaintSeconds, narWindow: r.narWindow, latentNorm: r.latentNorm,
                    memory: r.memory, lyrics: r.lyrics };
        if (r.probeRender) o.probeRender = true;
        if (r.how === 'bridge') { o.bridge = true; o.scores = Array.from(r.bridgeScores || []); }
        if (r.how === 'fresh') { o.fresh = true; o.trim = r.trim; o.phaseFit = r.phaseFit; }
        if (r.lyricLines) o.lyricLines = r.lyricLines;
        if (r.lyricWords) {
            o.lyricWords = Array.from(r.lyricWords);
            o.lyricWordLines = Array.from(r.lyricWordLines);
            o.pointer = Array.from(r.pointer);
            o.pointerMass = Array.from(r.pointerMass, (m) => Math.round(m * 1e5) / 1e5);
            o.linesReached = Array.from(r.linesReached);
        }
        return o;
    });
    writeJson(dir + '/chunks.json', {
        name: st.name, memory: st.memory, chunk: C, lookahead: opts.lookahead, narLeft: opts.narLeft,
        cot: st.cot || 'off', seed: opts.seed, odeSteps: opts.odeSteps, cfgScale: st.cfgScale, bridge: !!st.bridge,
        changeMemory: st.changeMemory, lyricsRoll: st.lyricsRoll, frames: K * C, audioFrames: a.frames, wall,
        source: 'yue.endless', programSamples: emitted, programContiguous: contiguous,
        stats: { work: s.stats.work, worstGapSeconds: s.stats.worstGapSeconds, costSeconds: s.stats.costSeconds },
        events, schedule: st.schedule, chunks,
    });
    console.log(st.name + ': ' + K + ' chunks, ' + wall.toFixed(1) + ' s wall; program ' + (emitted / 48000).toFixed(2) +
                ' s ' + (contiguous ? 'contiguous' : 'NOT contiguous') + '; norm ' +
                rows.map((r) => r.latentNorm.toFixed(1)).join(' '));
}
console.log('done');

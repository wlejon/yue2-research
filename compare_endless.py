"""Compare a yue.endless run (native/endless_stream.js) with an endless.js run.

    python compare_endless.py runs/endless_check/const_folk_m250 runs/exp5/const_folk_m250

Reports, per chunk, whether the tokens are identical, the latents' max
abs difference, and the audio's; then the row fields both runs carry.
"""
import json
import sys

import numpy as np


def load(d):
    j = json.load(open(d + '/chunks.json'))
    C = j['chunk']
    return (j, C, np.fromfile(d + '/tokens.i32', np.int32), np.fromfile(d + '/latents.f32', np.float32).reshape(-1, 64),
            np.fromfile(d + '/audio.f32', np.float32))


def main(a, b):
    ja, C, ta, la, aa = load(a)
    jb, _, tb, lb, ab = load(b)
    K = min(len(ja['chunks']), len(jb['chunks']))
    tok_same = lat_same = 0
    for k in range(K):
        s = slice(k * C, (k + 1) * C)
        t = np.array_equal(ta[s], tb[s])
        d = float(np.abs(la[s] - lb[s]).max())
        ra, rb = ja['chunks'][k], jb['chunks'][k]
        notes = []
        for f in ('memory', 'narWindow', 'trim', 'bridge', 'fresh', 'probeRender'):
            if ra.get(f) != rb.get(f):
                notes.append('%s %r/%r' % (f, ra.get(f), rb.get(f)))
        if 'phaseFit' in ra or 'phaseFit' in rb:
            pa, pb = ra.get('phaseFit') or {}, rb.get('phaseFit') or {}
            for f in ('streamPhase', 'freshPhase', 'cut'):
                if pa.get(f) != pb.get(f):
                    notes.append('phase.%s %r/%r' % (f, pa.get(f), pb.get(f)))
        if 'scores' in ra or 'scores' in rb:
            sa, sb = np.array(ra.get('scores', [])), np.array(rb.get('scores', []))
            if sa.shape != sb.shape or (len(sa) and np.abs(sa - sb).max() > 1e-6):
                notes.append('scores %s/%s' % (sa, sb))
        if (ra.get('lyrics') or '') != (rb.get('lyrics') or ''):
            notes.append('lyrics differ')
        if ra.get('heardText') != rb.get('heardText'):
            notes.append('heard %r/%r' % (ra.get('heardText'), rb.get('heardText')))
        tok_same += t
        lat_same += d == 0
        print('%3d %-12s tokens %s  latents max|d| %.3g  %s' % (k, ra.get('style', ''), 'same' if t else 'DIFF', d,
                                                            '; '.join(notes)))
    n = min(len(aa), len(ab))
    print('tokens identical in %d/%d chunks, latents in %d/%d; audio max|d| %.3g over %d samples'
          % (tok_same, K, lat_same, K, float(np.abs(aa[:n] - ab[:n]).max()), n))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

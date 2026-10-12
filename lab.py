"""Reference-model harness (PyTorch, the `yue2` package from ../YuE2-3B/.venv).

Native brosoundml is the fast path and the timing oracle; this harness is for
what the native seams cannot do yet:
  * NAR windows with codec context on either side and latent inpainting of
    already-emitted frames (stream seams),
  * NAR-path residual steering at arbitrary (layer, eval, frame) sets,
  * AR continuation from an arbitrary prefix (re-prompting mid-song: a new
    [Tags] text over the codec tokens already generated).

    from lab import Lab
    lab = Lab()
    lat = lab.nar(prefix, codec, noise)                        # one chunk = protocol
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from yue2.protocol import (CODEC_OFFSET, MUSIC_END, SongRequest, Sampling, GenerationConfig,
                           token_prefixes, negative_prefix)
from yue2.nar import attention

MODEL_DIR = Path("D:/projects/YuE2-3B")
VAE_DIR = Path("D:/projects/brosoundml/weights/yue2-vae")


def song_noise(seed, frames):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn((frames, 64), dtype=torch.float32, device="cpu", generator=g)


def logit_t(t):
    return torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20).item()


class Lab:
    def __init__(self, device="cuda", load_vae=True):
        from yue2.modeling_yue2 import YuE2ForCausalLM
        from yue2.tokenization_yue2 import YuE2TextTokenizer
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device(device)
        t = time.perf_counter()
        self.model = YuE2ForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16,
                                                     low_cpu_mem_usage=True).eval().to(self.device)
        self.tok = YuE2TextTokenizer(MODEL_DIR / "qwen.tiktoken")
        self.vae = None
        if load_vae:
            from yue2.modeling_vae import YuE2VAE
            self.vae = YuE2VAE.from_pretrained(VAE_DIR, decoder_only=True, device=self.device, local_files_only=True)
        print(f"lab loaded in {time.perf_counter() - t:.1f} s")

    # ── prompts ─────────────────────────────────────────────────────────────
    def request(self, style, lyrics, cot="full", seed=831001, abc=None):
        return SongRequest(style=style, lyrics=lyrics, cot=cot, seed=seed, abc=abc)

    def prefix(self, style, lyrics, cot="full", abc_ids=None):
        req = SongRequest(style=style, lyrics=lyrics, cot=cot)
        return token_prefixes(req, self.tok, abc_ids)

    # ── NAR ─────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def _prefill(self, ar_tokens):
        backbone = self.model.model
        ids = torch.tensor([ar_tokens], dtype=torch.long, device=self.device)
        pos = torch.arange(len(ar_tokens), device=self.device)[None]
        cos, sin = backbone.rotary_emb(pos)
        x = backbone.embed_tokens(ids)
        cache = []
        for layer in backbone.layers:
            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
            cache.append((k[0], v[0]))
            h = attention(q[0], k[0], v[0], causal=True)
            x = x + layer.self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return cache

    @torch.inference_mode()
    def song_cache(self, prefix, codec):
        """AR K/V of prompt + the whole song's codec (no MUSIC_END). Causal, so
        rows [0, n) are exactly a prefill of the first n tokens."""
        return self._prefill(list(prefix) + [int(c) + CODEC_OFFSET for c in codec])

    @torch.inference_mode()
    def history_cache(self, song_cache, n):
        """The NAR's AR cache for prompt+codec rows [0, n) + MUSIC_END at
        position n, from a song cache: one single-token AR forward."""
        backbone = self.model.model
        x = backbone.embed_tokens(torch.tensor([[MUSIC_END]], device=self.device))
        cos, sin = backbone.rotary_emb(torch.tensor([[n]], device=self.device))
        out = []
        for layer, (k, v) in zip(backbone.layers, song_cache):
            q, k1, v1 = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
            K, V = torch.cat((k[:n], k1[0])), torch.cat((v[:n], v1[0]))
            out.append((K, V))
            h = attention(q[0], K, V)
            x = x + layer.self_attn.o_proj(h.flatten(1)[None])
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return out

    @torch.inference_mode()
    def nar(self, prefix, codec, noise, steps=32, known=None, known_mask=None, steer=None, capture=None,
            pos_offset=0, cache=None):
        """Flow-match latents for len(noise) frames conditioned on
        prefix + codec + MUSIC_END (codec = ids 0..32767).

        known/known_mask: frames to clamp to `known` (inpainting by
        replacement: before every velocity evaluation the clamped frames are
        set to the exact flow interpolant t*noise + (1-t)*known).
        steer: list of dict(layer, delta[2048] or [frames,2048] tensor, scale,
        evals=None|set, frames=None|bool mask) added after that layer's output.
        capture: dict(layers=[...], evals=[...]) -> filled with
        {(layer, eval): [frames, 2048] float32 cpu}.
        pos_offset: latent position-embedding offset (0 = the protocol).
        """
        m = self.model
        ar_tokens = list(prefix) + [int(c) + CODEC_OFFSET for c in codec] + [MUSIC_END]
        if cache is None:
            cache = self._prefill(ar_tokens)
        n = len(noise)
        L = len(ar_tokens)
        dtype = next(m.vae2llm.parameters()).dtype
        positions = torch.arange(L, L + n + 2, device=self.device)[None]
        cos, sin = m.model.rotary_emb(positions)
        local = (torch.arange(n + 2, device=self.device) + pos_offset).clamp(max=m.config.max_latent_frames - 1)
        pos_emb = m.latent_pos_embed(local)[None]
        noise_d = noise.to(self.device, dtype)
        if known is not None:
            known_d = torch.as_tensor(known).to(self.device, dtype)
            kmask = torch.as_tensor(known_mask, device=self.device)[:, None]
        eval_i = [0]

        def clamp(state, t):
            if known is None:
                return state
            return torch.where(kmask, t * noise_d + (1 - t) * known_d, state)

        def velocity(state, raw_t):
            e = eval_i[0]
            eval_i[0] += 1
            x_nar = F.pad(state, (0, 0, 1, 1))
            shifted = m._shift_t_value(raw_t, self.device, dtype)
            x = m.vae2llm(x_nar[None]) + m.time_embedder(shifted.expand(n + 2))[None] + pos_emb
            for li, (layer, (ak, av)) in enumerate(zip(m.model.layers, cache)):
                q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), cos, sin)
                k, v = torch.cat((ak, k[0])), torch.cat((av, v[0]))
                h = attention(q[0], k, v)
                x = x + layer.nar_self_attn.o_proj(h.flatten(1)[None])
                x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
                if steer:
                    for s in steer:
                        if s["layer"] != li or (s.get("evals") is not None and e not in s["evals"]):
                            continue
                        d = torch.as_tensor(s["delta"], device=self.device, dtype=x.dtype) * s.get("scale", 1.0)
                        if d.ndim == 1:
                            d = d[None].expand(n, -1)
                        if s.get("frames") is not None:
                            d = d * torch.as_tensor(s["frames"], device=self.device, dtype=x.dtype)[:, None]
                        x[0, 1:-1] += d
                if capture is not None and li in capture.get("layers", ()) and e in capture.get("evals", ()):
                    capture.setdefault("out", {})[(li, e)] = x[0, 1:-1].float().cpu()
            return m.llm2vae(m.model.norm(x))[0, 1:-1]

        state = clamp(noise_d.clone(), 1.0)
        dt = 1.0 / steps
        for step in range(steps):
            t = 1.0 - step * dt
            state = clamp(state, t)
            v1 = velocity(state, logit_t(t))
            mid = clamp(state - v1 * (dt / 2), t - dt / 2)
            state = state - velocity(mid, logit_t(t - dt / 2)) * dt
        state = clamp(state, 0.0)
        return state.float().cpu()

    def window(self, prefix, codec_all, noise_all, a, b, left=0, right=0, known_left=None, steps=32,
               ctx_codec=True, steer=None, history=False, abs_pos=False, future=False, scache=None):
        """Latents for core frames [a, b) of a song.

        The solve covers frames [a-left, b+right) with the song's own noise;
        with ctx_codec the AR prefix carries codec[a-left, b+right) (else only
        codec[a, b)). known_left: the already-emitted latents of frames
        [a-left, a), clamped (inpainting) when given.
        history: the AR prefix carries every codec token from the song's start
        (codec[0, b+right)), as the semantic stage's own KV cache holds them.
        future: the AR prefix carries the whole song's codec (upper bound).
        abs_pos: latent position embeddings continue from the solve's first
        frame instead of restarting at 0.
        """
        lo, hi = max(0, a - left), min(len(codec_all), b + right)
        codec = codec_all[lo:hi] if ctx_codec else codec_all[a:b]
        if history:
            codec = codec_all[:hi]
        if future:
            codec = codec_all
        noise = noise_all[lo:hi]
        known = mask = None
        if known_left is not None and a > lo:
            known = torch.zeros(hi - lo, 64)
            known[: a - lo] = torch.as_tensor(known_left[-(a - lo):])
            mask = torch.zeros(hi - lo, dtype=torch.bool)
            mask[: a - lo] = True
        if steer:
            steer = [dict(s, frames=None if s.get("frames") is None else s["frames"][lo:hi]) for s in steer]
        cache = None
        if scache is not None and (history or future):
            cache = self.history_cache(scache, len(prefix) + len(codec))
        lat = self.nar(prefix, codec, noise, steps=steps, known=known, known_mask=mask, steer=steer,
                       pos_offset=lo if abs_pos else 0, cache=cache)
        return lat[a - lo: b - lo]

    def stream(self, prefix, codec_all, noise_all, W, left=0, right=0, inpaint=False, steps=32, frames=None,
               ctx_codec=True, steer=None, history=False, abs_pos=False, future=False):
        T = frames or len(codec_all)
        scache = self.song_cache(prefix, codec_all) if (history or future) else None
        out = []
        # W: one size, or a list of sizes (the last repeats): growing windows
        sizes = list(W) if isinstance(W, (list, tuple)) else [W]
        bounds, a, i = [], 0, 0
        while a < T:
            bounds.append((a, min(T, a + sizes[min(i, len(sizes) - 1)])))
            a, i = bounds[-1][1], i + 1
        for a, b in bounds:
            prev = torch.cat(out) if out else None
            # history: codec_all is passed whole so right context past T exists; window() cuts it at b+right
            out.append(self.window(prefix, codec_all, noise_all, a, b, left, right,
                                   prev if inpaint else None, steps, ctx_codec, steer, history, abs_pos, future,
                                   scache))
        return torch.cat(out)

    # ── VAE ─────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def decode(self, latents):
        z = torch.as_tensor(np.asarray(latents), dtype=torch.float32).T[None]
        audio = self.vae.decode_tiled(z, core_frames=1024, halo_frames=16, output_device="cpu")
        return audio[0].float().numpy()            # [2, N]

    # ── AR continuation ─────────────────────────────────────────────────────
    def continue_semantic(self, prefix, codec_so_far, n, seed=1, sampling=None, negative=None, cfg=1.0):
        """Sample n more codec tokens after prefix + codec_so_far (any prefix:
        e.g. a re-styled prompt over the song's existing tokens)."""
        from yue2.sampling import generate_tokens
        s = sampling or Sampling(min_tokens=0, max_tokens=n)
        full = list(prefix) + [int(c) + CODEC_OFFSET for c in codec_so_far]
        neg = None if negative is None else list(negative) + [int(c) + CODEC_OFFSET for c in codec_so_far]
        ids, timing, _ = generate_tokens(self.model, full, s, seed, "semantic", negative=neg, cfg_scale=cfg)
        return [int(t) - CODEC_OFFSET for t in ids], timing


def save_run(d, latents=None, audio=None, semantic=None, plan=None, info=None):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    if latents is not None:
        np.asarray(latents, dtype=np.float32).tofile(d / "latents.f32")
    if audio is not None:
        np.asarray(audio, dtype=np.float32).tofile(d / "audio.f32")
    if semantic is not None:
        np.asarray(semantic, dtype=np.int32).tofile(d / "semantic.i32")
    if plan is not None:
        (d / "plan.json").write_text(json.dumps(plan, ensure_ascii=False))
    (d / "run.json").write_text(json.dumps(info or {}))

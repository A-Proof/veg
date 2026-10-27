#!/usr/bin/env python3
"""VEG-3: Veyrian Execution Generator - resonance stack.

No torch, no transformers, no softmax attention. Hand-derived backward pass on NumPy/CuPy.

Why VEG-3: in VEG-2 every routing quantity (payload, write gate, content key, phase, decay)
was a function of the current byte only, so memory affinity m[t,j] collapsed to a fixed
256x256 byte-pair table. VEG-3 computes routing from the context-carrying residual stream
and stacks L resonance layers, so routing at layer l uses everything layers < l inferred.

Per layer (residual stream u, width D, NH resonance heads):
  a      = RMSnorm(u) * g1
  theta  = a @ Wp + bp                                  phase angles   [NH x PP]
  coh    = mean_i cos(theta_t,i - theta_j,i)            phase resonance
  cg     = sigmoid(q_t . k_j / sqrt(RK))                content gate,  q,k = a@Wq, a@Wk
  m[t,j] = coh * cg / sqrt(t+1)            (j <= t)     resonance routing
  M      = m @ (a @ Wv)                                 associative read
  rt     = sigmoid((a @ Ur) @ Vr + br)                  selective decay, log-spaced horizons
  s_t    = rt*s_{t-1} + (1-rt)*v_t + c*M_t              phase field, linear-time scan
  u     += (RMSnorm(s) * gs) @ Wo
  u     += (swish(b@W1) * (b@W3)) @ W2,   b = RMSnorm(u) * g2
Readout: logits = (RMSnorm(u) * gf) @ Wout + bout over 256 bytes.
"""
import argparse, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if __package__:
    from . import safeveyra as sv, delta_memory
else:
    import safeveyra as sv
    import delta_memory
GPU = bool(os.environ.get('VEG_GPU'))
if GPU:
    import cupy as np
else:
    import numpy as np
import numpy as _np

DT = np.float64 if os.environ.get('VEG_F64') else np.float32
VOC = 256
ARCH = 'VEG-resonance-stack/3'
KEYS = ('D', 'L', 'NH', 'PP', 'RK', 'FF', 'UR')
SIZES = {
    'S': dict(D=256, L=4, NH=4, PP=32, RK=32, FF=704, UR=32),
    'M': dict(D=384, L=6, NH=6, PP=32, RK=32, FF=1024, UR=64),
    'L': dict(D=512, L=8, NH=8, PP=32, RK=32, FF=1408, UR=64),
    'XL': dict(D=768, L=10, NH=12, PP=32, RK=32, FF=2048, UR=96),
}


def sig(x):
    return 1 / (1 + np.exp(-x))


def init(c, T, seed=0):
    r = _np.random.RandomState(seed)
    D, L, NH, PP, RK, FF, UR = (c[k] for k in KEYS)
    nrm = lambda *s, std: r.randn(*s) * std
    tau = _np.exp(_np.linspace(_np.log(1.5), _np.log(2.0 * T), D // NH))
    br = _np.tile(_np.log(tau - 1), NH)                                  # logit(1 - 1/tau)
    so = (2 * L) ** -.5
    w = {'emb': nrm(VOC, D, std=1.0), 'gf': _np.ones(D),
         'Wout': nrm(D, VOC, std=D ** -.5), 'bout': _np.zeros(VOC)}
    for l in range(L):
        p = f'{l}.'
        w.update({p + 'g1': _np.ones(D),
                  p + 'Wp': nrm(D, NH * PP, std=2 * D ** -.5), p + 'bp': r.uniform(0, 2 * _np.pi, NH * PP),
                  p + 'Wq': nrm(D, NH * RK, std=D ** -.5), p + 'Wk': nrm(D, NH * RK, std=D ** -.5),
                  p + 'Wv': nrm(D, D, std=D ** -.5),
                  p + 'Ur': nrm(D, UR, std=D ** -.5), p + 'Vr': nrm(UR, D, std=.1 * UR ** -.5),
                  p + 'br': br.copy(), p + 'c': _np.ones(D),
                  p + 'gs': _np.ones(D), p + 'Wo': nrm(D, D, std=so * D ** -.5),
                  p + 'g2': _np.ones(D),
                  p + 'W1': nrm(D, FF, std=D ** -.5), p + 'W3': nrm(D, FF, std=D ** -.5),
                  p + 'W2': nrm(FF, D, std=so * FF ** -.5)})
    return {k: np.asarray(v, dtype=DT) for k, v in w.items()}


FUSE = GPU and not os.environ.get('VEG_NOFUSE')
_SRC = r'''
extern "C" __global__ void field_f(const float* pre, const float* br, const float* v, const float* M,
    const float* c, float* s, float* rt, int T, int D){
  int n = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x; if (d >= D) return;
  long o = (long)n * T * D + d; float st = 0.f, b = br[d], cc = c[d];
  for (int t = 0; t < T; t++) {
    long i = o + (long)t * D; float r = 1.f / (1.f + expf(-(pre[i] + b)));
    rt[i] = r; st = r * st + (1.f - r) * v[i] + cc * M[i]; s[i] = st; }
}
extern "C" __global__ void field_b(const float* ds, const float* rt, const float* s, const float* v,
    const float* M, const float* c, float* dpre, float* dv, float* dM, float* dcp, int T, int D){
  int n = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x; if (d >= D) return;
  long o = (long)n * T * D + d; float l = 0.f, acc = 0.f, cc = c[d];
  for (int t = T - 1; t >= 0; t--) {
    long i = o + (long)t * D; float rn = t < T - 1 ? rt[i + D] : 0.f; l = rn * l + ds[i];
    float r = rt[i], sp = t > 0 ? s[i - D] : 0.f;
    dpre[i] = l * (sp - v[i]) * r * (1.f - r); dv[i] = l * (1.f - r); dM[i] = l * cc; acc += l * M[i]; }
  dcp[(long)n * D + d] = acc;
}
'''
_K = {}


def _k(name):
    if name not in _K:
        if name in ('field_f', 'field_b'):
            _K[name] = np.RawKernel(_SRC, name)
        elif name == 'route_f':
            _K[name] = np.ElementwiseKernel(
                'float32 coh, float32 qk, int32 T, float32 sc', 'float32 m',
                'int j = i % T; int t = (i / T) % T;'
                'm = j <= t ? coh / (1.f + expf(-qk * sc)) * rsqrtf(t + 1.f) : 0.f;', 'veg_route_f')
        elif name == 'route_b':
            _K[name] = np.ElementwiseKernel(
                'float32 dmr, float32 coh, float32 qk, int32 T, float32 sc', 'float32 dlog, float32 dcoh',
                'int j = i % T; int t = (i / T) % T;'
                'if (j > t) { dlog = 0.f; dcoh = 0.f; } else {'
                ' float cg = 1.f / (1.f + expf(-qk * sc)), dd = dmr * rsqrtf(t + 1.f);'
                ' dcoh = dd * cg; dlog = dd * coh * cg * (1.f - cg) * sc; }', 'veg_route_b')
        elif name == 'glu_f':
            _K[name] = np.ElementwiseKernel('float32 h1, float32 h3', 'float32 o',
                                            'o = h1 / (1.f + expf(-h1)) * h3;', 'veg_glu_f')
        elif name == 'glu_b':
            _K[name] = np.ElementwiseKernel(
                'float32 dgl, float32 h1, float32 h3', 'float32 gl, float32 dh1, float32 dh3',
                'float s = 1.f / (1.f + expf(-h1)), sw = h1 * s;'
                'gl = sw * h3; dh3 = dgl * sw; dh1 = dgl * h3 * (s + h1 * s * (1.f - s));', 'veg_glu_b')
    return _K[name]


def _scan(inp, r, back):
    N, T, D = inp.shape
    out = np.empty_like(inp); st = np.zeros_like(inp[:, 0])
    for t in (range(T - 1, -1, -1) if back else range(T)):
        st = r[:, t] * st + inp[:, t]; out[:, t] = st
    return out


_MK = {}


def _mask(T):
    if T not in _MK:
        _MK[T] = np.tril(np.ones((T, T), DT)) / np.sqrt(np.arange(1, T + 1, dtype=DT))[:, None]
    return _MK[T]


def route_f(coh, qk, sc):
    if FUSE:
        return _k('route_f')(coh, qk, np.int32(coh.shape[-1]), np.float32(sc))
    return coh * sig(qk * sc) * _mask(coh.shape[-1])


def route_b(dmr, coh, qk, sc):
    if FUSE:
        return _k('route_b')(dmr, coh, qk, np.int32(coh.shape[-1]), np.float32(sc))
    cg = sig(qk * sc); dd = dmr * _mask(coh.shape[-1])
    return dd * coh * cg * (1 - cg) * sc, dd * cg


def glu_f(h1, h3):
    return _k('glu_f')(h1, h3) if FUSE else h1 * sig(h1) * h3


def glu_b(dgl, h1, h3):
    if FUSE:
        return _k('glu_b')(dgl, h1, h3)
    s1 = sig(h1); sw = h1 * s1
    return sw * h3, dgl * h3 * (s1 + h1 * s1 * (1 - s1)), dgl * sw


def _grid(N, D):
    return (N, (D + 255) // 256), (256,)


def field_f(pre, br, v, M, c):
    if FUSE:
        N, T, D = v.shape
        s = np.empty_like(v); rt = np.empty_like(v)
        _k('field_f')(*_grid(N, D), (np.ascontiguousarray(pre), br, np.ascontiguousarray(v),
                                     np.ascontiguousarray(M), c, s, rt, np.int32(T), np.int32(D)))
        return s, rt
    rt = sig(pre + br)
    return _scan((1 - rt) * v + c * M, rt, 0), rt


def field_b(ds, rt, s, v, M, c):
    N, T, D = v.shape
    if FUSE:
        dpre, dv, dM = np.empty_like(v), np.empty_like(v), np.empty_like(v)
        dcp = np.empty((N, D), v.dtype)
        _k('field_b')(*_grid(N, D), (np.ascontiguousarray(ds), rt, s, v, M, c, dpre, dv, dM, dcp,
                                     np.int32(T), np.int32(D)))
        return dpre, dv, dM, dcp.sum(0)
    z = np.zeros((N, 1, D), v.dtype)
    lam = _scan(ds, np.concatenate([rt[:, 1:], z], 1), 1)
    dpre = lam * (np.concatenate([z, s[:, :-1]], 1) - v) * rt * (1 - rt)
    return dpre, lam * (1 - rt), lam * c, (lam * M).sum((0, 1))


def rms(x, g):
    r = np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)
    return x / r * g, r


def rms_b(dy, x, r, g):
    xn = x / r
    dxn = dy * g
    return (dxn - xn * (dxn * xn).mean(-1, keepdims=True)) / r, (dy * xn).sum((0, 1))


def wg(a, d):
    return a.reshape(-1, a.shape[-1]).T @ d.reshape(-1, d.shape[-1])


def split(x, h):
    N, T, C = x.shape
    return x.reshape(N, T, h, C // h).transpose(0, 2, 1, 3)


def merge(x):
    N, h, T, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(N, T, h * d)


def tr(x):
    return x.swapaxes(-1, -2)


def layer_f(w, c, l, u, mask):
    p = f'{l}.'; NH, PP, RK = c['NH'], c['PP'], c['RK']
    a, r1 = rms(u, w[p + 'g1'])
    th = a @ w[p + 'Wp'] + w[p + 'bp']
    F = np.concatenate([split(np.sin(th), NH), split(np.cos(th), NH)], -1) * PP ** -.5
    q = split(a @ w[p + 'Wq'], NH); k = split(a @ w[p + 'Wk'], NH)
    v = a @ w[p + 'Wv']
    if c.get('memory', 'resonance') == 'delta':
        mh, memory_cache = delta_memory.forward(np, F, q, k, split(v, NH))
        coh, qk = memory_cache, None
        M = merge(mh)
    else:
        coh = F @ tr(F)
        qk = q @ tr(k)
        M = merge(route_f(coh, qk, RK ** -.5) @ split(v, NH))
    hr = a @ w[p + 'Ur']
    s, rt = field_f(hr @ w[p + 'Vr'], w[p + 'br'], v, M, w[p + 'c'])
    sn, r2 = rms(s, w[p + 'gs'])
    u1 = u + sn @ w[p + 'Wo']
    b, r3 = rms(u1, w[p + 'g2'])
    h1 = b @ w[p + 'W1']; h3 = b @ w[p + 'W3']
    u2 = u1 + glu_f(h1, h3) @ w[p + 'W2']
    return u2, (u, a, r1, th, F, q, k, qk, coh, v, M, hr, rt, s, sn, r2, u1, b, r3, h1, h3)


def layer_b(w, c, l, du2, ch, mask, g):
    p = f'{l}.'; NH, PP, RK = c['NH'], c['PP'], c['RK']
    u, a, r1, th, F, q, k, qk, coh, v, M, hr, rt, s, sn, r2, u1, b, r3, h1, h3 = ch
    gl, dh1, dh3 = glu_b(du2 @ w[p + 'W2'].T, h1, h3)
    g[p + 'W2'] = wg(gl, du2)
    g[p + 'W1'] = wg(b, dh1); g[p + 'W3'] = wg(b, dh3)
    dx, g[p + 'g2'] = rms_b(dh1 @ w[p + 'W1'].T + dh3 @ w[p + 'W3'].T, u1, r3, w[p + 'g2'])
    du1 = du2 + dx
    g[p + 'Wo'] = wg(sn, du1)
    ds, g[p + 'gs'] = rms_b(du1 @ w[p + 'Wo'].T, s, r2, w[p + 'gs'])
    dpre, dv, dM, g[p + 'c'] = field_b(ds, rt, s, v, M, w[p + 'c'])
    g[p + 'br'] = dpre.sum((0, 1))
    g[p + 'Vr'] = wg(hr, dpre)
    dhr = dpre @ w[p + 'Vr'].T
    g[p + 'Ur'] = wg(a, dhr)
    da = dhr @ w[p + 'Ur'].T
    dMh = split(dM, NH)
    if c.get('memory', 'resonance') == 'delta':
        dF, dqh, dkh, dvh = delta_memory.backward(np, dMh, coh)
        dq, dk = merge(dqh), merge(dkh)
        dv = dv + merge(dvh)
    else:
        dv = dv + merge(tr(route_f(coh, qk, RK ** -.5)) @ dMh)
        dlog, dcoh = route_b(dMh @ tr(split(v, NH)), coh, qk, RK ** -.5)
        dq = merge(dlog @ k); dk = merge(tr(dlog) @ q)
        dF = dcoh @ F + tr(dcoh) @ F
    g[p + 'Wv'] = wg(a, dv)
    da += dv @ w[p + 'Wv'].T
    g[p + 'Wq'] = wg(a, dq); g[p + 'Wk'] = wg(a, dk)
    da += dq @ w[p + 'Wq'].T + dk @ w[p + 'Wk'].T
    dF = dF * PP ** -.5
    dth = merge(dF[..., :PP]) * np.cos(th) - merge(dF[..., PP:]) * np.sin(th)
    g[p + 'Wp'] = wg(a, dth); g[p + 'bp'] = dth.sum((0, 1))
    da += dth @ w[p + 'Wp'].T
    dx, g[p + 'g1'] = rms_b(da, u, r1, w[p + 'g1'])
    return du1 + dx


def run(w, c, x, y=None):
    """y given: (nll, grads). y None: log-probs [N,T,256]."""
    N, T = x.shape
    mask = None if c.get('memory', 'resonance') == 'delta' else _mask(T)
    u = w['emb'][x]
    chs = []
    for l in range(c['L']):
        u, ch = layer_f(w, c, l, u, mask)
        if y is not None:
            chs.append(ch)
    hf, rf = rms(u, w['gf'])
    lo = hf @ w['Wout'] + w['bout']
    lo -= lo.max(-1, keepdims=True)
    ex = np.exp(lo); se = ex.sum(-1, keepdims=True)
    if y is None:
        return lo - np.log(se)
    ii = np.arange(N)[:, None]; jj = np.arange(T)[None]
    nll = -(lo[ii, jj, y] - np.log(se[..., 0])).mean()
    dl = ex / se
    dl[ii, jj, y] -= 1
    dl /= N * T
    g = {'Wout': wg(hf, dl), 'bout': dl.sum((0, 1))}
    du, g['gf'] = rms_b(dl @ w['Wout'].T, u, rf, w['gf'])
    for l in reversed(range(c['L'])):
        du = layer_b(w, c, l, du, chs.pop(), mask, g)
    oh = np.zeros((N * T, VOC), du.dtype)
    oh[np.arange(N * T), x.reshape(-1)] = 1
    g['emb'] = oh.T @ du.reshape(N * T, -1)
    return nll, g


def adamw(w, g, st, t, lr, b1=.9, b2=.95, wd=.1, clip=1.0):
    gn = float(np.sqrt(sum((gk * gk).sum() for gk in g.values())))
    if not math.isfinite(gn):
        return gn
    sc = min(1.0, clip / (gn + 1e-9))
    c1 = 1 - b1 ** t; c2 = 1 - b2 ** t
    for k, gk in g.items():
        m, v = st[k]
        gk = gk * sc
        m *= b1; m += (1 - b1) * gk
        v *= b2; v += (1 - b2) * gk * gk
        if w[k].ndim == 2 and k != 'emb':
            w[k] *= 1 - lr * wd
        w[k] -= lr * (m / c1) / (np.sqrt(v / c2) + 1e-8)
    return gn


def save(w, c, path, extra=None):
    host = {k: (v.get() if GPU else v) for k, v in w.items()}
    meta = {'format': sv.FMT, 'arch': ARCH, **c, **(extra or {})}
    sv.save_tensors(host, path + '.tmp', meta)
    os.replace(path + '.tmp', path)


def load(path):
    t, meta = sv.load_tensors(path)
    c = {k: int(meta[k]) for k in KEYS}
    c['memory'] = meta.get('memory', 'resonance')
    return {k: np.asarray(v, dtype=DT) for k, v in t.items()}, c


def evaluate(w, c, vd, T, n=256, bs=16):
    starts = _np.linspace(0, len(vd) - T - 2, n).astype(_np.int64)
    tot = 0.0
    for i in range(0, n, bs):
        sb = starts[i:i + bs]
        xx = np.asarray(_np.stack([vd[s:s + T] for s in sb]).astype(_np.int64))
        yy = np.asarray(_np.stack([vd[s + 1:s + T + 1] for s in sb]).astype(_np.int64))
        lp = run(w, c, xx)
        tot += float(-lp[np.arange(len(sb))[:, None], np.arange(T)[None], yy].sum())
    return tot / (n * T) / math.log(2)


def generate(w, c, prompt, n, temp, T, seed=0):
    buf = list(prompt.encode()); rng = _np.random.RandomState(seed)
    for _ in range(n):
        lp = run(w, c, np.asarray(_np.array(buf[-T:], dtype=_np.int64)[None]))[0, -1] / temp
        pr = _np.asarray(lp.get() if GPU else lp, dtype=_np.float64)
        pr = _np.exp(pr - pr.max()); pr /= pr.sum()
        buf.append(int(rng.choice(VOC, p=pr)))
    return bytes(buf).decode('utf-8', 'replace')


def gradcheck(memory='resonance'):
    c = dict(D=16, L=2, NH=2, PP=4, RK=4, FF=24, UR=4); T = 12
    c['memory'] = memory
    w = init(c, T, 1)
    rng = _np.random.RandomState(0)
    x = np.asarray(rng.randint(0, VOC, (2, T))); y = np.asarray(rng.randint(0, VOC, (2, T)))
    _, g = run(w, c, x, y)
    worst = 0.0
    for k in w:
        errs = []
        for _ in range(3):
            i = tuple(rng.randint(0, s) for s in w[k].shape)
            if k == 'emb':
                i = (int(x[0, rng.randint(T)]),) + i[1:]
            old = float(w[k][i]); eps = 1e-5
            w[k][i] = old + eps; lp, _ = run(w, c, x, y)
            w[k][i] = old - eps; lm, _ = run(w, c, x, y)
            w[k][i] = old
            num = (float(lp) - float(lm)) / (2 * eps); ana = float(g[k][i])
            errs.append(abs(num - ana) / max(1e-7, abs(num) + abs(ana)))
        worst = max(worst, max(errs))
        print(f'{k:8s} max rel err {max(errs):.2e}')
    print('WORST', f'{worst:.2e}')
    if worst > 1e-5:
        raise AssertionError('Gradient check failed; use VEG_F64=1 on CPU.')


def fusecheck():
    global FUSE
    c = dict(SIZES['S'], L=2); T = 128
    w = init(c, T, 1); rng = _np.random.RandomState(0)
    x = np.asarray(rng.randint(0, VOC, (4, T))); y = np.asarray(rng.randint(0, VOC, (4, T)))
    FUSE = True; n1, g1 = run(w, c, x, y)
    FUSE = False; n2, g2 = run(w, c, x, y)
    print('nll fused', float(n1), 'ref', float(n2))
    print('WORST rel grad diff', max(float(np.linalg.norm(g1[k] - g2[k]) / (np.linalg.norm(g2[k]) + 1e-12)) for k in g2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', default='M', choices=list(SIZES))
    ap.add_argument('--memory', choices=('resonance', 'delta'), default=None)
    for k in KEYS:
        ap.add_argument('--' + k, type=int, default=0)
    ap.add_argument('--T', type=int, default=512)
    ap.add_argument('--data', default='train.bin'); ap.add_argument('--vdata', default='val.bin')
    ap.add_argument('--rows', type=int, default=32); ap.add_argument('--accum', type=int, default=1)
    ap.add_argument('--steps', type=int, default=1000); ap.add_argument('--hours', type=float, default=0)
    ap.add_argument('--lr', type=float, default=1.5e-3); ap.add_argument('--lr-final', type=float, default=0)
    ap.add_argument('--warmup', type=int, default=200); ap.add_argument('--wd', type=float, default=.1)
    ap.add_argument('--eval-every', type=int, default=1000); ap.add_argument('--eval-n', type=int, default=256)
    ap.add_argument('--save', default=''); ap.add_argument('--load', default='')
    ap.add_argument('--gen', default=''); ap.add_argument('--genlen', type=int, default=300)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--gradcheck', action='store_true'); ap.add_argument('--bench', type=int, default=0)
    ap.add_argument('--fusecheck', action='store_true')
    ap.add_argument('--to-safetensors', nargs=2, metavar=('SRC', 'DST'))
    a = ap.parse_args()
    if a.gradcheck:
        return gradcheck(a.memory or 'delta')
    if a.fusecheck:
        return fusecheck()
    if a.to_safetensors:
        return sv.to_safetensors(*a.to_safetensors)
    T = a.T
    if a.load:
        w, c = load(a.load)
    else:
        c = dict(SIZES[a.size]); c.update({k: getattr(a, k) for k in KEYS if getattr(a, k)})
        c['memory'] = 'delta'
        w = init(c, T)
    if a.memory is not None:
        c['memory'] = a.memory
    print('config', c, 'params', sum(int(v.size) for v in w.values()), flush=True)
    vd = _np.memmap(a.vdata, _np.uint8, 'r')
    if a.gen and not a.steps:
        print(generate(w, c, a.gen, a.genlen, a.temp, T)); return
    srcs = [_np.memmap(f, _np.uint8, 'r') for f in a.data.split(',')]
    wts = _np.array([len(s) for s in srcs], dtype=_np.float64); wts /= wts.sum()
    rng = _np.random.RandomState(1)
    ost = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in w.items()}
    steps, lrf = a.steps, a.lr_final or a.lr * .1
    best, t0, tl, loss_acc, nacc = 9e9, time.time(), time.time(), 0.0, 0
    st = 0
    while st < steps:
        st += 1
        if st <= a.warmup:
            lr = a.lr * st / a.warmup
        else:
            lr = lrf + (a.lr - lrf) * .5 * (1 + math.cos(math.pi * min(1.0, (st - a.warmup) / max(1, steps - a.warmup))))
        acc = None
        for _ in range(a.accum):
            src = srcs[rng.choice(len(srcs), p=wts)]
            s0 = rng.randint(0, len(src) - T - 1, a.rows)
            xy = _np.stack([src[s:s + T + 1] for s in s0]).astype(_np.int64)
            nll, g = run(w, c, np.asarray(xy[:, :-1]), np.asarray(xy[:, 1:]))
            loss_acc += nll; nacc += 1
            if acc is None:
                acc = g
            else:
                for k in acc:
                    acc[k] += g[k]
        if a.accum > 1:
            for k in acc:
                acc[k] /= a.accum
        gn = adamw(w, acc, ost, st, lr, wd=a.wd)
        if a.bench and st == a.bench:
            el = time.time() - t0
            print(f'BENCH {a.bench} steps {el:.1f}s  {el / a.bench * 1000:.0f} ms/step  '
                  f'{a.bench * a.rows * a.accum * T / el / 1e3:.1f} KB/s', flush=True); return
        if a.hours and (st == 40 or (st > 40 and st % 500 == 0)):
            rate = (time.time() - tl) / (st - sl)
            steps = max(st + 1, int(st + (a.hours * 3600 * .95 - (time.time() - t0)) / rate))
            print(f'calibrated {rate * 1000:.0f} ms/step -> {steps} total steps', flush=True)
            tl, sl = time.time(), st
        if st == 10:
            tl, sl = time.time(), st
        if st % 50 == 0:
            el = time.time() - t0
            print(f'step {st}/{steps} bpb {float(loss_acc) / nacc / math.log(2):.4f} lr {lr:.2e} '
                  f'gn {gn:.2f} {st * a.rows * a.accum * T / el / 1e3:.1f}KB/s {el / 3600:.2f}h', flush=True)
            loss_acc, nacc = 0.0, 0
        if st % a.eval_every == 0 or st == steps:
            vb = evaluate(w, c, vd, T, a.eval_n)
            tag = ''
            if a.save:
                save(w, c, a.save, {'step': st, 'val_bpb': f'{vb:.4f}'})
                if vb < best:
                    best = vb; save(w, c, a.save.replace('.safeveyra', '.best.safeveyra'), {'step': st, 'val_bpb': f'{vb:.4f}'})
                    tag = ' *best'
            print(f'VAL step {st} bpb {vb:.4f}{tag}', flush=True)
            if a.gen and st < steps:
                print('SAMPLE', repr(generate(w, c, a.gen, 160, a.temp, T)), flush=True)
    if a.gen:
        print('SAMPLE:', generate(w, c, a.gen, a.genlen, a.temp, T), flush=True)


if __name__ == '__main__':
    main()

"""Reference prediction-error associative memory, with analytical gradients.

Inputs use [batch, heads, time, channels]. No learned parameters are added.
The implementation caches states for BPTT; it is not the production GPU kernel.
"""
import os

try:
    if __package__:
        from . import _native
    else:
        try:
            import _native
        except ImportError:
            from veg import _native
except ImportError:
    _native = None


def normalize(np, x):
    scale = np.sqrt((x * x).sum(-1, keepdims=True) + 1e-6)
    return x / scale, scale


def forward(np, phase, q, k, v):
    """Read after a gated delta-rule write; current input is causally visible."""
    rank = q.shape[-1]
    tq, tk = np.tanh(q), np.tanh(k)
    query, qs = normalize(np, np.concatenate([phase, tq / rank ** .5], -1))
    key, ks = normalize(np, np.concatenate([phase, tk / rank ** .5], -1))
    beta = 1 / (1 + np.exp(-k.mean(-1, keepdims=True)))
    state = np.zeros((*v.shape[:2], key.shape[-1], v.shape[-1]), dtype=v.dtype)
    out = np.empty_like(v)
    history = []
    if (_native is not None and np.__name__ == 'numpy'
            and not os.environ.get('VEG_NONATIVE') and v.dtype.name in ('float32', 'float64')):
        states = np.empty((*v.shape[:3], key.shape[-1], v.shape[-1]), dtype=v.dtype)
        errors = np.empty_like(v, order='C')
        out = np.empty_like(v, order='C')
        _native.delta_forward(*(np.ascontiguousarray(a) for a in (query, key, beta, v)),
                              out, states, errors, state)
        history = [(states[:, :, t], errors[:, :, t]) for t in range(v.shape[2])]
    else:
        for t in range(v.shape[2]):
            kt, qt = key[:, :, t], query[:, :, t]
            error = v[:, :, t] - (kt[..., None] * state).sum(-2)
            history.append((state, error))
            state = state + kt[..., None] * (beta[:, :, t] * error)[..., None, :]
            out[:, :, t] = (qt[..., None] * state).sum(-2)
    return out, (phase, tq, tk, query, key, qs, ks, beta, history, state)


def backward(np, dy, cache):
    """Return gradients for phase, q, k and v, in that order."""
    phase, tq, tk, query, key, qs, ks, beta, history, state = cache
    dq, dk, dv = np.zeros_like(query), np.zeros_like(key), np.zeros_like(dy)
    db = np.zeros_like(beta)
    ds = np.zeros_like(state)
    for t in reversed(range(dy.shape[2])):
        before, error = history[t]
        kt, qt, bt = key[:, :, t], query[:, :, t], beta[:, :, t]
        after = before + kt[..., None] * (bt * error)[..., None, :]
        dyt = dy[:, :, t]
        dq[:, :, t] = (after * dyt[..., None, :]).sum(-1)
        ds = ds + qt[..., None] * dyt[..., None, :]
        dup = (ds * kt[..., None]).sum(-2)
        dk[:, :, t] = (ds * (bt * error)[..., None, :]).sum(-1)
        db[:, :, t] = (dup * error).sum(-1, keepdims=True)
        de = dup * bt
        dv[:, :, t] = de
        dk[:, :, t] -= (before * de[..., None, :]).sum(-1)
        ds = ds - kt[..., None] * de[..., None, :]
    dq = (dq - query * (dq * query).sum(-1, keepdims=True)) / qs
    dk = (dk - key * (dk * key).sum(-1, keepdims=True)) / ks
    width, rank = phase.shape[-1], tq.shape[-1]
    dphase = dq[..., :width] + dk[..., :width]
    dquery = dq[..., width:] * (1 - tq * tq) / rank ** .5
    dkey = dk[..., width:] * (1 - tk * tk) / rank ** .5
    dkey += db * beta * (1 - beta) / rank
    return dphase, dquery, dkey, dv

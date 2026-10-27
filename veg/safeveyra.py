#!/usr/bin/env python3
""".safeveyra <-> .safetensors serialization for VEG.

.safeveyra uses the safetensors container layout (8-byte LE header length,
JSON tensor index, raw fp32 payloads) and adds a __metadata__ section that
describes the VEG architecture version and dimensions. Because the container
is already safetensors-compatible, conversion to .safetensors is a lossless
re-header: same tensors, metadata reduced to a provenance note.

No dependencies beyond numpy.
"""
import json, struct
import numpy as np

MAGIC = b'VEYRA\x00\x01\x00\x00'          # format tag stored inside __metadata__ instead of file magic
FMT = 'safeveyra/1'
ARCH = 'VEG-phase-field/2'

_DT = {'F32': np.float32, 'F16': np.float16, 'BF16': 'bf16', 'I64': np.int64, 'U8': np.uint8}


def save_tensors(tensors, path, meta=None):
    """Write a safetensors-layout file. tensors: {name: np.ndarray}."""
    header = {}
    blob = b''
    off = 0
    for name, t in tensors.items():
        t = np.ascontiguousarray(np.asarray(t, dtype=np.float32))
        raw = t.tobytes()
        header[name] = {'dtype': 'F32', 'shape': list(t.shape), 'data_offsets': [off, off + len(raw)]}
        blob += raw
        off += len(raw)
    if meta:
        header['__metadata__'] = {k: str(v) for k, v in meta.items()}
    hj = json.dumps(header).encode()
    hj += b' ' * (-len(hj) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(hj)))
        f.write(hj)
        f.write(blob)


def load_tensors(path):
    """Read a safetensors-layout file -> (tensors dict, metadata dict)."""
    with open(path, 'rb') as f:
        (hlen,) = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(hlen))
        blob = f.read()
    meta = header.pop('__metadata__', {})
    tensors = {}
    for name, info in header.items():
        dt = _DT[info['dtype']]
        assert dt == np.float32, 'veg weights are fp32'
        a, b = info['data_offsets']
        tensors[name] = np.frombuffer(blob[a:b], dtype=np.float32).reshape(info['shape']).copy()
    return tensors, meta


def save_veg(w, path, extra_meta=None):
    """Save VEG weights as .safeveyra."""
    npz = {}
    for k, v in w.items():
        vv = v.get() if hasattr(v, 'get') else v        # cupy -> host
        npz[k] = np.asarray(vv, dtype=np.float32)
    meta = {'format': FMT, 'arch': ARCH,
            'P': w['om'].shape[0], 'H': w['Va'].shape[1],
            'H2': w['rho2'].shape[0], 'RK': w['Q'].shape[1], 'VOC': w['bias'].shape[0]}
    if extra_meta:
        meta.update(extra_meta)
    save_tensors(npz, path, meta)


def load_veg(path):
    """Load .safeveyra or .npz weights -> dict of fp32 arrays."""
    if path.endswith('.npz'):
        zf = np.load(path)
        return {k: zf[k].astype(np.float32) for k in zf.files}
    tensors, meta = load_tensors(path)
    return tensors


def to_safetensors(src, dst):
    """Convert .safeveyra -> .safetensors (lossless; keeps a provenance note)."""
    tensors, meta = load_tensors(src)
    note = {**meta, 'format': 'safetensors', 'arch': meta.get('arch', ARCH),
            'source': 'safeveyra',
            'veg_dims': json.dumps({k: meta.get(k) for k in ('P', 'H', 'H2', 'RK', 'VOC')})}
    save_tensors(tensors, dst, note)


def from_safetensors(src, dst):
    """Convert .safetensors -> .safeveyra (re-attaches VEG metadata)."""
    tensors, meta = load_tensors(src)
    save_tensors(tensors, dst, {**meta, 'format': FMT, 'arch': meta.get('arch', ARCH)})

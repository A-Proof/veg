# VEG: Veyrian Execution Generator

Experimental byte-level text generation in NumPy/CuPy, with hand-derived
gradients. No PyTorch or Transformers runtime dependency.

## Architecture

VEG combines sine/cosine phase features, selective recurrent fields, residual
connections and gated feed-forward computation. New models default to
prediction-error associative memory: each write corrects what memory already
predicts instead of simply accumulating another value.

The delta rule is an established recurrent-memory technique, not a claim of a
new learning rule. Legacy resonance mode uses attention-like pairwise routing.
These are research models, not validated frontier systems or brain simulations.

## Install

```sh
# Once published on PyPI:
python -m pip install veg-train veg-infer
# Both automatically install veg-core==0.2.0.

# From a source checkout (available now):
python -m pip install . ./packages/train ./packages/infer
# Optional CUDA 12 GPU backend for core:
python -m pip install '.[gpu]'
```

`veg-core` provides the Python `veg` library. `veg-train` and `veg-infer` provide
their respective terminal commands and depend on the exact core release.
These names replace the initial, unpublished `veyrian-veg` distribution.

An optional C extension fuses the CPU delta-memory forward scan and releases
the Python GIL. Source installations need a C compiler to enable it; otherwise
NumPy is used. Set `VEG_NONATIVE=1` to compare with the reference path. The
backward pass remains NumPy, so this is not a complete native trainer or CUDA
speedup. Tests compare C outputs and analytical gradients with the reference
in float32 and float64. No faster statistical learning is established.

### Publishing (Maintainers)

The `Publish To PyPI` GitHub Actions workflow builds, tests and publishes via
PyPI Trusted Publishing (no API token stored in GitHub).
For the first release, register three pending publishers at
https://pypi.org/manage/account/publishing/ with:

- PyPI projects (one registration each): `veg-core`, `veg-train`, `veg-infer`
- GitHub owner: `A-Proof`
- Repository: `veg`
- Workflow filename: `publish.yml`
- Environment: `pypi`

Then run the workflow manually from GitHub Actions, or publish a GitHub release.
Later releases must increment `version` in `pyproject.toml`; PyPI versions
cannot be overwritten. Keep all three project versions and their exact core
dependencies in sync. PyPI account setup requires the project owner's login.
The workflow uploads all three distributions; none is published until PyPI
authorization is configured.

## Train

Supply your own UTF-8 byte corpora, split by document into train and validation.
Data and model checkpoints are not distributed in this repository.

```sh
veg-train --size S --rows 1 --steps 100 --data train.bin \
  --vdata val.bin --save experiment.safeveyra
```

Delta memory currently uses a sequential correctness-reference implementation
with a large BPTT state cache. Start small; it is not yet a fast GPU trainer.
Use `--memory resonance` for the older GPU-optimized implementation.
Set `VEG_GPU=1` before starting Python to select CuPy.

Existing checkpoint metadata determines memory mode on loading; checkpoints
without a memory tag retain legacy resonance behavior. Explicitly specifying
`--memory delta` with an old checkpoint changes its computation, not just format.

## Python API

```python
import numpy as np
from veg import SIZES, init, run

config = dict(SIZES["S"], memory="delta")
weights = init(config, T=32)
log_probs = run(weights, config, np.array([[72, 101, 108, 108, 111]]))
```

## Checkpoints And Export

`.safeveyra` uses a safetensors-layout container with VEG metadata.
Exporting tensors does not convert VEG into a Transformer or provide
`AutoModel` compatibility; execution still requires VEG code.

```sh
veg-train --to-safetensors experiment.safeveyra experiment.safetensors
veg-train --load experiment.safeveyra --steps 0 --gen "The "
veg-infer --load experiment.safeveyra --prompt "The " --length 200
```

`veg-infer` needs only a checkpoint, not training or validation data.

Checkpoint saving currently persists weights/configuration, not optimizer or
random-generator state: loading is a warm start, not exact training resumption.

## Verification

```sh
cd veg
python -m unittest test_delta_memory
VEG_F64=1 python veg3.py --memory delta --gradcheck
VEG_F64=1 python veg3.py --memory resonance --gradcheck
```

See `veg/DELTA_MEMORY.md` for memory equations and matched comparison protocol.
The previous 25.45M-parameter resonance run reported 1.3669 validation bits/byte
on a local FineWeb-Edu split. That number does not evaluate delta memory and is
not a general intelligence benchmark or evidence of superiority to matched
Transformers. No delta-memory quality improvement has yet been established.

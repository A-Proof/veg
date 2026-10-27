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
python -m pip install .
# Optional CUDA 12 GPU backend:
python -m pip install '.[gpu]'
```

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
```

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

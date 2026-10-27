# Prediction-Error Memory Experiment

This default for new VEG-3 models changes memory, not parameter count or data.
Existing resonance checkpoints retain their original mode and are not modified.

For each head, normalized sine/cosine phase features are concatenated with
contextual query/key features. The existing key projection also supplies a
sigmoid write gate. There are no additional learned tensors.

At each position:

    prediction = key @ state
    error = value - prediction
    state += outer(key, gate * error)
    output = query @ state

This is delta-rule associative memory, a known family of recurrent mechanisms,
not a claim of a newly discovered learning rule. Current-input writes are
visible to the current output; future inputs are not.

## Verification

Run on CPU with NumPy:

    python -m unittest test_delta_memory
    VEG_F64=1 python veg3.py --memory delta --gradcheck
    VEG_F64=1 python veg3.py --memory resonance --gradcheck

## Comparison

Use identical dimensions, initialization seed, corpus, sequence length and
byte exposure for both variants, then compare validation bpb and wall time.
Also compare at equal wall time, since equal exposure is not equal compute.
Select on validation and evaluate the final choice on an untouched test split.
Switching memory on an existing checkpoint changes its semantics; a warm-start
comparison needs the same source checkpoint for both arms.

    python veg3.py --size S --memory delta --steps 400 --rows 8
    python veg3.py --size S --memory resonance --steps 400 --rows 8

The delta path caches memory states for analytical BPTT. An optional C
extension fuses the CPU forward loop, with a NumPy fallback; backward remains
NumPy/CuPy. Forward work is linear in sequence length
for fixed dimensions, but training cache memory is O(batch * heads * time *
feature_width * value_width). It may be slower and use more memory than the
existing GPU implementation. A fused/chunked implementation is required before
a large overnight run. No quality improvement or 1.08 bpb result is established.

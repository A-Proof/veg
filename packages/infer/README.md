# veg-infer

Inference command for the experimental Veyrian Execution Generator.
Automatically installs the matching `veg-core` version.

    python -m pip install veg-infer
    veg-infer --load model.safeveyra --prompt "The " --length 200

You must supply a trained VEG checkpoint. No model weights are bundled.
No training or validation corpus is required for inference.

Source and documentation: https://github.com/A-Proof/veg

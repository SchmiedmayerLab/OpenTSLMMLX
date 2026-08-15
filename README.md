<!--

This source file is part of the OpenTSLM MLX open-source project

SPDX-FileCopyrightText: 2026 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)

SPDX-License-Identifier: MIT

-->

# OpenTSLM SP — MLX

[![Build and Test](https://github.com/SchmiedmayerLab/OpenTSLMMLX/actions/workflows/static-analysis.yml/badge.svg)](https://github.com/SchmiedmayerLab/OpenTSLMMLX/actions/workflows/static-analysis.yml)
[![REUSE status](https://api.reuse.software/badge/github.com/SchmiedmayerLab/OpenTSLMMLX)](https://api.reuse.software/info/github.com/SchmiedmayerLab/OpenTSLMMLX)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

MLX port of [OpenTSLM](https://github.com/SchmiedmayerLab/OpenTSLM)'s SP (Soft Prompt) variant for inference on Apple Silicon.

## Project Structure

```
src/                  # Python MLX implementation
  ts_encoder.py       # TransformerCNNEncoder
  ts_projector.py     # MLPProjector
  opentslm_sp.py      # End-to-end model (includes interleave logic)
  sleep_dataset.py    # Sleep-EDF dataset loader (auto-downloads data)
checkpoints/          # Converted safetensors weights
models/               # LLM base weights
```

## Setup

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the base LLM

Download [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) in **bf16** (full precision) and place it in `models/`:

```bash
hf download meta-llama/Llama-3.2-1B --local-dir models/Llama-3.2-1B-bf16
```

The bf16 model is required because the LoRA adapters in the checkpoints were trained against
full-precision weights. Quantized models (e.g. 4-bit) have different weight shapes and cannot
be combined with these LoRA weights.

### 3. Download a checkpoint

Download a trained `.pt` checkpoint and place it in `checkpoints/`:

| Checkpoint            | Task                             | Source                                                      |
| --------------------- | -------------------------------- | ----------------------------------------------------------- |
| `model_checkpoint.pt` | EEG / sleep stage classification | [HuggingFace](https://huggingface.co/OpenTSLM) |

Then run one-time conversion to safetensors:

```bash
# One-time conversion dependencies
pip install torch peft safetensors

python convert_checkpoint.py \
  --input checkpoints/model_checkpoint.pt \
  --output-prefix checkpoints/model_checkpoint
```

This writes:

- `checkpoints/model_checkpoint.encoder.safetensors`
- `checkpoints/model_checkpoint.projector.safetensors`
- `checkpoints/model_checkpoint.lora.safetensors` (if LoRA exists)

## Running Inference

The Sleep-EDF dataset is auto-downloaded on first run.

```bash
source .venv/bin/activate

# Single sample from Sleep-EDF test set
python inference.py

# Pick a specific sample
python inference.py --sample-idx 5

# Control generation length
python inference.py --max-new-tokens 500
```

## Contributing

Contributions to this project are welcome. Please make sure to read the [contribution guidelines](https://github.com/SchmiedmayerLab/.github/blob/main/CONTRIBUTING.md) and the [contributor covenant code of conduct](https://github.com/SchmiedmayerLab/.github/blob/main/CODE_OF_CONDUCT.md) first. You can find a list of contributors in the [CONTRIBUTORS.md](CONTRIBUTORS.md) file.

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for more information.

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](CITATION.cff), which GitHub surfaces through the [*Cite this repository*](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) button.

## Our Research

For more information, visit the [Schmiedmayer Lab GitHub organization](https://github.com/SchmiedmayerLab).

![Schmiedmayer Lab](https://raw.githubusercontent.com/SchmiedmayerLab/.github/main/assets/footer-light.png#gh-light-mode-only)
![Schmiedmayer Lab](https://raw.githubusercontent.com/SchmiedmayerLab/.github/main/assets/footer-dark.png#gh-dark-mode-only)

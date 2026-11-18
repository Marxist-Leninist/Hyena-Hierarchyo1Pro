# Hyena Hierarchy: legacy experiments and a working educational LM

The original `h6.py`, `hyena.py` and `hyenakalilinuxversion.py` are preserved unchanged.
The old documentation is preserved in [docs/legacy-readme.md](docs/legacy-readme.md).

**New recommended entry point: `hyena_lm.py`.** This is an educational
Hyena-inspired character model, **not** a reproduction of the paper, an official
Hyena implementation, or a pretrained general-purpose LLM.

## What changed

The legacy Kali script feeds numeric character IDs into linear/convolutional layers;
without nonlinearities or input-dependent gates that composition is affine.
It does not implement the long-convolution/gating mechanism described in
[Hyena Hierarchy (Poli et al., 2023)](https://arxiv.org/abs/2302.10866).
The [authors' implementation](https://github.com/HazyResearch/safari) remains the
reference for research reproduction.

The new independent implementation provides:
- Learned character embeddings and per-token next-character targets.
- Causal depthwise short convolutions, implicit position-conditioned long filters,
  zero-padded FFT linear convolution, and input-dependent multiplicative gates.
- Residual blocks, layer normalization and feed-forward nonlinearities.
- A deterministic sorted vocabulary, strict unknown-character errors, and
  generation from short or over-context prompts.
- Finite training runs, gradient clipping, fixed disjoint validation windows,
  atomic checkpoints, optimizer/RNG state and dataset hashes.
- Eight tests, including causality under future-token changes and prefix lengths,
  direct-versus-FFT convolution, gradient flow, and resumed optimizer updates.

The filter MLP uses SiLU and a small fixed Fourier basis, rather than attempting
to copy all paper hyperparameters, initialization and optimized kernels. This
version has no incremental inference cache, custom CUDA kernels, mixed precision,
or demonstrated speed/quality advantage over another architecture.

## Run (Python 3.10+, PyTorch 2.6+)

Install PyTorch for your platform using its official installation instructions.
Only PyTorch and the Python standard library are required by the new files.

```bash
python -m unittest -v test_hyena_lm
python make_demo_data.py
python hyena_lm.py train --text examples/synthetic.txt --context 32 --width 16 --layers 1 --steps 200 --batch-size 8 --threads 2 --save-every 50 --output checkpoints/synthetic-demo.pt
python hyena_lm.py generate --checkpoint checkpoints/synthetic-demo.pt --prompt "the " --length 100
python hyena_lm.py train --text examples/synthetic.txt --resume checkpoints/synthetic-demo.pt --steps 100 --batch-size 8 --output checkpoints/continued.pt
```

Use your own UTF-8 text with `--text`. The last 10% of characters forms validation;
neither split may be shorter than context + 1. Windows do not cross the split.
Vocabulary is derived from the full corpus (including validation characters),
which is disclosed here rather than represented as an unseen-vocabulary test.
Case is preserved. Set `--device cuda` to explicitly select a supported CUDA device;
the tested configuration is CPU float32.

Resume restores architecture, vocabulary, optimizer (including its learning rate)
and RNG states. Supply the identical dataset and batch size for same-setting
continuation. `--steps` means **additional** steps. Architecture flags are ignored
when resuming. Each invocation writes its own metrics file beside its output,
so use a new output filename to retain previous run logs. CPU RNG, sampling RNG
and available CUDA RNG states are saved; cross-device/library-version bitwise
reproducibility is not promised.

Checkpoints are loaded with `weights_only=True`. Use only trusted files anyway.
A save on interruption preserves current state, but an interruption during an
optimizer update is not guaranteed to be an exact replay boundary. No dataset
text or filesystem path is embedded in the checkpoint; it includes a SHA-256
dataset fingerprint and basic training metadata.

## Measured smoke result, 2026-11-18

CPU, PyTorch 2.10.0, 2 threads, seed 42; 3,911 parameters; 200 AdamW updates.
The generated toy corpus contains a repeated synthetic grammar, with overlapping
templates in both disjoint splits. This is a **pipeline/learning sanity check**,
not evidence of linguistic competence, generalization, or benchmark improvement.

| Update | Fixed validation cross-entropy (nats/character) |
|---:|---:|
| 0 | 3.262248 |
| 50 | 2.501134 |
| 100 | 1.914615 |
| 150 | 1.393656 |
| 200 | 0.965810 |

The greedy output is still repetitive and malformed. Eight functional tests pass;
a two-update checkpoint continuation also succeeded. Only bounded CPU tests were
performed. No private chats, external text datasets, paid compute or large downloads
were used. See [results/synthetic-demo.metrics.json](results/synthetic-demo.metrics.json).

## Provenance

This repository is a fork of Suro-One/Hyena-Hierarchy. Historical files retain
their provenance and are not relicensed here. The new implementation is an
independent educational expression of the long-convolution/gating idea, not copied
upstream source. No repository-wide license grant is inferred from a public fork.

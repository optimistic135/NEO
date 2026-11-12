# CAP-NEO on CIFAR-10-C

The original `main.py` and `tta_library/neo.py` remain available. The CAP runner calls the original NEO forward on the exact same frozen ViT class-token features used by the new methods. It retains the repository's CIFAR-10 fine-tuned ViT-Base and ViT preprocessing configuration, pinned to Hugging Face revisions in `config.json`. The runner reads the original CIFAR-10-C arrays or the repository's converted `corruption/severity/0…9` image folders.

## Server commands

Run from the repository root with the CIFAR-10-C dataset on a GPU node. For the existing `df-atta` environment with PyTorch 2.1.2.post304, use Transformers 4.51.3. The constraint pins the currently installed Torch during dependency resolution.

```bash
conda activate df-atta
bash <<'BASH'
set -euo pipefail
torch_version="$(python -c 'from importlib.metadata import version; print(version("torch"))')"
python -m pip install "transformers==4.51.3" \
  --constraint <(printf 'torch==%s\n' "$torch_version")
BASH
```

For a clean Python 3.11 environment with the repository's PyTorch 2.7.1, use `python -m pip install -r requirements.txt`. Both routes use the existing repository dependencies including NumPy, SciPy, Pillow, Matplotlib and tqdm. The first run downloads the pinned model and processor to the Hugging Face cache. Use a shared writable `HF_HOME` when downloading from a login node for an offline compute node.

```bash
bash <<'BASH'
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-$HOME/.cache/neo-huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.cache/neo-matplotlib}"
DATA=/data/user3/chenc/datasets/CIFAR-10-C
RUNS="$PWD/cap_results/viT-base-cifar10c-l5"

python -m unittest discover -s tests -p 'test_cap_*.py'

# Stage 1: one-corruption check, including batch-by-batch Softmax equivalence.
python -u run_cap_neo.py --stage baseline --data "$DATA" \
  --output "$RUNS/smoke" --corruptions gaussian_noise --seeds 2020 --device cuda

# Stage 1: original NEO on the full natural streams.
python -u run_cap_neo.py --stage baseline --data "$DATA" --output "$RUNS/baseline" --device cuda

# Stage 2: four methods on identical natural streams.
python -u run_cap_neo.py --stage ablation --data "$DATA" --output "$RUNS/ablation" --device cuda

# Stage 3: four label-free methods on six controlled adaptation priors.
python -u run_cap_neo.py --stage imbalance --data "$DATA" --output "$RUNS/imbalance" --device cuda

# Stage 4: offline label oracle on both natural and controlled streams.
python -u run_cap_neo.py --stage oracle --data "$DATA" --output "$RUNS/oracle" --device cuda

python report_cap_neo.py \
  --runs "$RUNS/baseline" "$RUNS/ablation" "$RUNS/imbalance" "$RUNS/oracle" \
  --output "$RUNS/report"
BASH
```

Stage 1 can also be compared with existing original-entry-point JSONs by appending `--reference-dir /path/to/main.py/results` to the reporting command. Those legacy files do not record sample order or IDs. The comparison labels their split identity as unverified. The report checks the new baseline and ablation NEO metrics against one another on identical ordered splits. The README's published CIFAR-10-C number is context for the reproduction and is not an acceptance target.

To generate fresh legacy JSONs for the comparison, use a dedicated working directory and give `main.py` the prepared ImageFolder path. The original loader needs PNG or JPEG class folders.

```bash
bash <<'BASH'
set -euo pipefail
repo="$PWD"
# Replace this with the existing converted images directory.
prepared=/absolute/path/to/cifar10-c-l5-neo512-run/images
test -d "$prepared/gaussian_noise/5"
legacy="$(mktemp -d "$repo/cap-legacy-XXXXXX")"
cd "$legacy"
for seed in 1234 2020 9999; do
  CUDA_VISIBLE_DEVICES=0 python -u "$repo/main.py" --cifar_10 \
    --data_cifar_10_c "$prepared" --algorithm neo --vit_type base \
    --corruption all --level 5 --seed "$seed" --adapt_num_samples 512 \
    --eval --batch_size 64 --workers 0 2>&1 | tee "seed-$seed.log"
done
printf 'Legacy JSON directory %s\n' "$legacy/results"
BASH
```

The CAP runner defaults to workers 0. Each stage displays progress by seed and corruption and by batch. An interrupted run can be continued with the same command and output directory. Completed stream JSONs are reused after checking the saved configuration and split manifest. Use distinct output directories for concurrent jobs or changed configurations.

## Protocol

- Default configuration: 15 corruptions, severity 5, seeds 1234, 2020 and 9999, batch size 64, 512 adaptation images per corruption. Reset all state before each stream.
- Natural stream: use the original seeded random partition into 512 adaptation and 9,488 validation images. Materialize an independently seeded order for each split. This removes dependence on model initialization, prior corruptions and logging from the order used for comparisons.
- Controlled streams: draw without replacement. Choose dominant class `seed % 10` and distribute the remaining slots as evenly as possible over the nine other classes. Report both target and achieved dominant shares. At 10%, distribute all slots as evenly as integer counts allow. At 100%, adaptation contains only the dominant class.
- Controlled validation: each view contains 480 held-out images. The balanced view contains 48 per class and is fixed across all ratios for a seed and corruption. The matched view applies the target adaptation ratio to its own counts. The two views can overlap and are identical at 10%. A shared frozen forward over their union avoids duplicate feature extraction. Both views are disjoint from adaptation.
- Adaptation metrics use each batch's logits after updating from that batch, matching NEO's online scoring time. Validation uses the final correction throughout. Labels in formal methods reach only the sampler or scoring code. Oracle labels reach only the separate oracle stage and only during adaptation.
- The backbone and classifier weights are frozen. Each batch goes through the backbone once per stream, shared by all selected methods. Predictions and metrics consume each batch before the next arrives. The bank contains the feature vectors and raw logits from all adaptation batches seen so far.

## Estimator and ablations

For centered classifier weights `V = A = PW`, recompute `p_i = softmax(W h_i + b - q_prev)` over the complete bank at each update. Form `z_i = p_i V`. Fit a single nonnegative scale by regressing centered features on centered prototypes. Subtract the mean of the eligible fit subset separately from the features and prototypes before computing the scale.

```text
kappa = max(0, sum(dot(z_centered, h_centered)) / sum(||z_centered||²))
```

Use maximum corrected softmax probability of at least 0.7, at least 16 eligible fit samples, at least two supported classes with at least two samples each, and centered cosine alignment of at least 0.05. These defaults are configurable and are hypotheses to validate. Insufficient coverage, degenerate fit energy, nonpositive scale or weak alignment give κ = 0. The residual shift uses the full bank, including samples outside the fit subset.

```text
r_i = h_i - kappa * z_i
a_i = A @ r_i
q_hat = mean(a_i)
s² = sum(||a_i - q_hat||²) / (N * (N - 1))
lambda = clip(1 - s² / (||q_hat||² + eps), 0, 1)
q = lambda * q_hat
corrected_logits = W @ h + b - q
```

The shrinkage estimate assumes residual variability across the bank is informative about estimation error. Accuracy changes are empirical. With a singleton bank, shrinkage uses λ = 0. If variance or signal is nonfinite, use zero shrinkage. If the candidate correction cannot produce finite logits, retain the previous finite state when usable, otherwise revert to zero. Invalid backbone outputs fail with an explicit error. At κ = 0 and λ = 1, the correction differs from original NEO by a common class-logit offset, so their Softmax predictions coincide.

| Method | Class composition | Discriminative-space shrinkage |
| --- | --- | --- |
| `neo` | Raw feature mean | Original NEO forward |
| `neo_shrink` | κ = 0 | Enabled |
| `cap_pa` | Soft predicted prior | λ = 1 |
| `cap_neo` | Soft predicted prior | Enabled |
| `cap_oracle_labels` | Adaptation label one-hot prior | Enabled |

A single true class leaves the oracle's centered prototype fit unidentified. The formal methods can only detect insufficient *predicted* coverage. A confidently misclassified single-class stream can still appear to have multiple classes, so inspect estimated prior error and fallback frequency together. The label oracle probes the prototype model and κ estimator, and the classifier's geometry can still limit its benefit. No clean-pair shift metric is generated by this runner. Its configuration explicitly marks that optional diagnostic as skipped.

## Files and interpretation

Each stage writes `config.json`, `run.log`, `per_corruption.csv`, and one JSON per seed/corruption/stream under `runs/`. Stream JSONs include the ordered indices and stable sample IDs, class counts for every split, and κ, λ, correction norms, priors and fallback reasons per update. The CSV includes top-1/top-5 in percent, entropy in nats, 15-bin ECE in [0, 1], final diagnostics, and time and peak memory in seconds and MiB. Prior total variation compares the estimated soft prior with the true adaptation distribution **after** evaluation.

`shared_backbone_seconds` includes transfer of the image batch to the selected device and extraction. `method_seconds` times the method's adaptation and prediction calls with GPU synchronization. It includes the diagnostics recorded inside CAP updates. Original NEO diagnostics and the stage-1 equivalence check are recorded separately from that method's timing. `inference_seconds` sums shared backbone and method times. `stream_wall_seconds` covers the complete stream, including loading, metrics and diagnostics. `peak_gpu_mib` is the peak allocated GPU memory observed during the shared backbone and that method's calls. It includes co-resident states when methods are run together. `method_peak_extra_gpu_mib` measures additional allocations above memory resident immediately before each method call. For isolated GPU peaks, repeat `--stage ablation --methods <method>` in a separate output directory per method. CPU runs leave GPU memory fields blank.

The report writes overall and per-corruption tables, paired ablation deltas, oracle comparisons and κ/λ diagnostics. It also writes imbalance and diagnostic plots as PNG and SVG. Overall metrics are averaged over corruptions within each seed before computing the mean and sample standard deviation over seeds. Accuracy deltas are in percentage points. Use `method_comparison.csv` for natural-stream method results and `ablation_summary.csv` for paired differences. If CAP-NEO loses accuracy, inspect soft-prior total variation, κ alignment and fallback counts in the stream JSONs, and compare κ/λ with paired accuracy changes before attributing the difference to a module.

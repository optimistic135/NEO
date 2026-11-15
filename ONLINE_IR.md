# PLEAD-Lite and Online IR

Run in the same GPU environment used by PLEAD-V2. Each line runs one dataset from the repository root. The ImageNet-C profile covers all 15 corruptions, including `zoom_blur`.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 HF_HOME="${HF_HOME:-$HOME/.cache/neo-huggingface}" python -u run_online_ir.py --dataset cifar10-c --data /data/user3/chenc/datasets/CIFAR-10-C --output results/online_ir/cifar10-c
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 HF_HOME="${HF_HOME:-$HOME/.cache/neo-huggingface}" python -u run_online_ir.py --dataset cifar100-c --data /data/user3/chenc/datasets/CIFAR-100-C --output results/online_ir/cifar100-c
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 HF_HOME="${HF_HOME:-$HOME/.cache/neo-huggingface}" python -u run_online_ir.py --dataset imagenet-c --data /data/user3/chenc/datasets/ImageNet-C --output results/online_ir/imagenet-c
```

Use `--smoke` with a separate output directory to evaluate the first two subsets per IR under Gaussian noise, severity 5 and seed 2020. The full profiles use severity 5 and seeds 1234, 2020 and 9999. Completed stream files are written atomically. An interrupted run can resume with the same configuration and output path. The result directory lock prevents a second writer. A restarted process rebuilds its in-memory feature cache on the first unfinished stream for each corruption.

Merge the completed datasets into a single report:

```bash
python -m cap_experiments.ir_reporting --runs results/online_ir/cifar10-c results/online_ir/cifar100-c results/online_ir/imagenet-c --output results/online_ir
```

| Dataset | Subsets × images | Batch size | FIFO length | IR |
| --- | --- | ---: | ---: | --- |
| CIFAR-10-C | 10 × 1,000 | 64 | 128 | 1, 5, 20, 50, 5000 |
| CIFAR-100-C | 100 × 100 | 64 | 100 | 1, 200, 500, 50000 |
| ImageNet-C | 1,000 × 50 | 50 | 50 | 1, 1000, 5000, 500000 |

The diagonal quota rounds the nominal dominant mass to the nearest integer, with ties rounded up. Off-diagonal quotas differ by at most one in each subset. The remainder rotates in the same class ranking so every class has the same global count and every input appears once per complete stream. Seeded permutations determine class order, subset order and intra-subset order. The sampler fails if the population differs from the specified capacity. In particular, very high IR may round to an entirely single-class subset. At ImageNet-C IR=1, a 50-image subset can represent only 50 of 1,000 classes. The report includes nominal IR, actual dominant percentages, nonzero class counts, and sparse histograms represented as pairs of class ID and count. The ImageNet-C variant uses 50 unique images per subset. The DART paper uses 100. [DART Appendix A.1](https://arxiv.org/html/2411.15204v1#A1.SS1)

Each batch is scored after updating from arrivals in that batch, consistent with the existing batch-transductive evaluation. Strict prequential scoring predicts before updating. Batches stop at subset boundaries, and both adapters keep state across them. In ImageNet-C a new 50-image batch evicts the prior FIFO contents, while Lite carries the preceding correction across the boundary. Adaptation methods receive only features or the shared FIFO, with no access to labels, dominance, or IR.

Lite stores detached features, uncorrected logits and projected features in the FIFO. It recomputes probabilities from the current correction each time. The full and interleaved-half ridge fits use batched tensor operations. The gate emits independent gamma and lambda diagnostics. A full fit failure gives zero debiasing with shrinkage on the direct mean residual. NEO-Online applies the original affine head to features centered by the same FIFO mean.

A separate arrival cache lets independent streams for the same corruption reuse a frozen model output. It is filled only when an image arrives. A stream reads exactly the current batch IDs and starts with a fresh FIFO and correction. The main experiment reports warm-cache reads and actual backbone calls. Shared feature and window construction times are charged to both methods. Method time is the adapter update on the existing window. Peak GPU memory includes the shared feature cache and co-resident state. Use the separate efficiency benchmark to compare the cold V2 Fit+Gate and Lite pipelines.

The efficiency benchmark uses the first two Gaussian-noise IR=1 subsets, warm-up for both methods, and three synchronized timed repeats. It compares logits batchwise at the same FIFO length. End-to-end inference includes transfer of preprocessed inputs, one backbone forward per batch, cache construction, and adaptation. CPU image decode is outside the timed region. The Lite window-build field is measured separately from cache read plus update. The old V2 method time includes its internal cache construction and diagnostics.

Each dataset exports the requested CSVs, booktabs tables, and plots. Per-stream compressed JSON contains the full index sequence and SHA-256, per-batch fit diagnostics and FIFO label histograms for evaluation. The output also includes `dominant_class_summary.csv` and `efficiency_table.tex`. Summaries average corruptions within seed, then report mean and sample standard deviation. The paired Top-1 interval uses a Student t interval over seed-level means of paired corruption differences. Smoke intervals remain blank because smoke uses one seed.

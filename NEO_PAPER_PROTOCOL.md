# NEO paper protocol experiments

The paper reports several protocols for different research questions. The default runner reproduces the evaluation design of Table 2 with the original repository NEO adapter, and evaluates PLEAD (Single-View) on the very same input batches. The frozen classifier supplies a No Adapt control. The runner reuses the existing ViT-Base checkpoints and preprocessing. [NEO paper, §4.5 and §5.1](https://arxiv.org/html/2510.05635)

| Protocol | Adaptation data and state | Top-1 population | Paper context |
| --- | --- | --- | --- |
| `table2` | 512 random distinct images per corruption, eight 64-image batches; reset before each corruption | The same 512 images, update then score each batch | Table 2 |
| `heldout-512` | Same 512-image adaptation split and resets | Report adaptation accuracy and separately score all remaining images with the final state frozen | Appendix C.2, Table 4(a), 64-image setting |
| `continual` | 512 images per corruption, 64 per batch, randomly ordered corruptions with state preserved across boundaries | Each current batch, after updating | Figure 5(c) and Appendix C.8 |

`table2` uses the original sample-count-weighted running mean through all 512 images. PLEAD uses its existing equations with a 512-image bank, so no feature is evicted during an episodic run. In the continual option the repository's `NEO_Cont` applies EMA, with the author repository's default alpha=0.1 recorded as a run setting. PLEAD carries its 512-image FIFO and correction state across corruptions. This is a continual PLEAD extension. The original paper's few-sample and class-count studies in Figure 5(a–b) use different held-out protocols and are outside these three presets.

The main protocol evaluates all 15 standard corruptions at severity 5, including Zoom blur, with fixed seeds 1234, 2020 and 9999. ImageNet-C and CIFAR-10-C use the paper's data and adaptation/scoring design. The published runs' ordered sample manifests and exact seed list were unavailable, so matching the paper's numerical results exactly is not required. Sample IDs and ordering in the new runs are recorded. CIFAR-100-C and PLEAD are extensions evaluated under the same design. The author-reported Table 2 NEO reference accuracies are 59.2% for ImageNet-C and 82.4% for CIFAR-10-C.

Online IR instead changes the class proportions between subsets, evaluates the full stream, and places both methods on a bounded recent-sample window of 128/100/50 images. The Table 2 experiment uses sampling from each corruption's natural population, a fixed 512-image sample budget, and original cumulative NEO.

Run from the NEO repository root in the same GPU environment:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 HF_HOME="${HF_HOME:-$HOME/.cache/neo-huggingface}" python -u run_neo_paper.py --dataset cifar10-c --data /data/user3/chenc/datasets/CIFAR-10-C --output results/neo_paper/table2/cifar10-c
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 HF_HOME="${HF_HOME:-$HOME/.cache/neo-huggingface}" python -u run_neo_paper.py --dataset cifar100-c --data /data/user3/chenc/datasets/CIFAR-100-C --output results/neo_paper/table2/cifar100-c
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 HF_HOME="${HF_HOME:-$HOME/.cache/neo-huggingface}" python -u run_neo_paper.py --dataset imagenet-c --data /data/user3/chenc/datasets/ImageNet-C --output results/neo_paper/table2/imagenet-c
```

Use `--protocol heldout-512` or `--protocol continual` with its own output directory. A `--smoke` run uses seed 2020 with Gaussian and shot noise. Outputs include per-stream and summary CSVs, paired Top-1 deltas, Markdown tables, and ordered sample manifests. They also record Top-5, entropy, ECE, fallback, method and shared inference timings. Completed episodic streams can be resumed individually. A continual sequence is saved atomically per seed because its state depends on the full prefix.

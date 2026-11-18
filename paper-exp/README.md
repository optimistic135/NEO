# PLEAD Cross-Architecture TTA Benchmark

Run with the existing `df-atta` environment. Set `HF_HOME` to the NEO cache and `HF_HUB_OFFLINE=1`. All official baseline pins are in `configs/methods/official.yaml`. Clone PEA and ROID under `.upstream/pea` and `.upstream/roid` at those commits. PEA consumes the official alignment implementation and ROID consumes the official augmentation implementation. TENT, SAR and T3A have separate local adapters. T3A uses the published K=20 grid value fixed before evaluation. TENT uses the DART Appendix A.4 optimizer settings. SAR uses the official model-specific LR and top-layer exclusions, including Hugging Face aliases, scales the entropy margin by class count, and skips empty reliable sets. ROID applies its augmentation in pixel space and retains the official consistency objective, EMA and score correction. NEO applies the original cumulative center formula to the frozen penultimate features. PLEAD uses the repository's simplified method, with all current arrivals scored and the latest 128 used for adaptation when a batch exceeds the window.

```bash
python -u bench.py validate
python -m unittest discover -s tests
python -u bench.py smoke
python -u bench.py balanced
python -u bench.py online_ir --dart-manifests /path/to/verified/indices
python -u bench.py ablation
python -u bench.py ablation --online --dart-manifests /path/to/verified/indices
python -u bench.py efficiency
python -u bench.py report
```

`--datasets` and `--architectures` shard GPU runs. Successful stream records resume when saved checkpoint and batch hashes match. Benchmark timing uses single-method FP32 runs, five warm-up batches, reset, then 30 measured Gaussian Noise batches. PEA requires clean reference statistics from the same frozen checkpoint. Configure a local clean-data root via `pea_source`. The source loader discards class labels.

## DART inputs

[DART §2, A.1 and G.5](https://openreview.net/pdf?id=c7AAHdEYz5) specifies subset sizes, target probabilities and random subset ordering. It cites Niu et al. for the benchmark construction. The [official SAR generator](https://github.com/mr-eggplant/SAR/blob/20f6e24b17525f34503510afccedc0629b67b7c4/dataset/generate_shifted_sample_indices.py) samples per-position categories and samples images with replacement within a class. We could verify this generator for ImageNet, and could not authenticate a DART author implementation for all three datasets. The strict runner accepts an externally authenticated author index export at `dataset/seed_IR.npy`, shaped [classes, subset size], with a sibling JSON carrying `author_source`, `indices_sha256`, `sample_ids_sha256` and `dominant_class_order`. The sample ID digest uses compact JSON and SHA256. ImageNet IDs are paths relative to the configured corruption root. CIFAR IDs are severity-local array offsets. Sampling provenance is stored next to each saved stream. Online batches honor subset boundaries and keep state at boundaries. Confirm this batching convention when author artifacts become available. ViT DART is N/A. ResNet DART is Pending Reproduction.

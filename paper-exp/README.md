# CIFAR-100-C · ViT-B/16 · balanced-512

新入口为 `scripts/run_cifar100c.sh`。它使用实验 YAML 组合模型、CIFAR-C 数据、采样协议和每个方法的配置。这里只新增 CIFAR-100-C、ViT-B/16 和 balanced corruption。原来的 `bench.py` 是历史入口，保留供旧结果复现。

## 协议

- 15 种标准扰动，severity 5，种子 1234、2020、9999，batch size 64。
- 直接共用 balanced-512 划分函数。每流无放回分层采样 512 张，余下 9,488 张为独立验证集，适应与验证索引及 SHA256 保存一次供所有方法共用。
- 每条流从同一只读 checkpoint 新建方法状态。适应调用只接收图像，分类标签只用于分层和计分。
- 适应精度按各方法原生批预测时序计分。验证采用最终模型、模板、教师或 prompt/shift。LAME 没有持久适应状态，验证用独立批内图求解。
- 报告汇总每个种子的 15 个 corruption 后计算三种子的样本标准差。配对 95% CI 使用三个种子的配对差值（df=2）。

## 方法审核

`configs/methods/*.yaml` 固定论文、官方仓库 commit、配置与架构适配差异。TENT、T3A、SAR 复用已核对的框架适配器。官方 TENT 的 CIFAR 示例使用 BatchNorm；本实验使用 ViT LayerNorm affine 和 CIFAR Adam 设置。T3A 使用已有特征模板机制及固定的 K=20，避免用验证标签选 K。SAR 保留两次筛选与 SAM，空筛选时安全跳过。

CoTTA、DeYO、LAME、FOA 和 FOZO 直接调用固定提交的官方算法文件。运行时在 `third_party/` 缓存并校验 SHA256。CoTTA 迁移到 ViT 输入尺寸及归一化，保留教师/增强/参数恢复。DeYO 按 100 类缩放熵阈值，使用官方 PLPD 筛选和 patch shuffle。LAME 复用官方 affinity 和迭代求解器。FOA/FOZO 在冻结的同一个 Hugging Face ViT 上接入原生 prompt 流程。它们的离线源域统计从完整干净 CIFAR-100 训练集计算，缓存键跟踪权重、预处理、源文件及实现 SHA256。CIFAR-100-C 图像只用于目标适应与评分。

## 手动启动

```bash
CUDA_VISIBLE_DEVICES=0 bash paper-exp/scripts/run_cifar100c.sh
```

默认使用同一工作树的 balanced-512 Python 环境；可设置 `PAPER_EXP_PYTHON` 使用另一工作树中已安装的环境。Python 依赖通过清华 PyPI 安装。首次访问五个官方核心文件需要 GitHub 网络。模型按固定 Hugging Face 缓存读取，全程校验模型和预处理文件。指定 `--validate` 仅做模型与数据核验；`--methods tent sar` 选择方法；`--pilot` 只执行 smoke 和首种 corruption 的三种子试跑。通过 gate 的方法才会进入其余 14 种。再次提交同一命令复用已成功的逐流结果。配置变化时请选新的 `--output`。

输出在 `paper-exp/results/cifar100c_balanced_vitb/`，包括逐流和逐批数据、split、检查结果、模型/运行配置、来源校验、报告、CSV 和 PDF 图。历史实验结果路径独立。

---

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

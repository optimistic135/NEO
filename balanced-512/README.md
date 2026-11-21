# 6.1 类别均衡场景

该目录比较 ViT-B/16 上的 PEA 和 PLEAD。沿用 NEO 仓库已固定的三个 checkpoint 和预处理，评估 CIFAR-10-C、CIFAR-100-C、ImageNet-C 的 15 种标准扰动，severity 5，种子 1234、2020、9999。每个 corruption 和种子重新初始化在线状态。

每类先取等额样本，余数由种子确定的类别排列分配。适应集共 512 张且无重复，剩余 9,488 张 CIFAR-C 或 49,488 张 ImageNet-C 作为验证集。适应和验证各自打乱，batch size 64。标签只用于分层划分和计算指标。适应 Top-1 来自当前批适应后的预测，主表使用冻结后的独立验证 Top-1。报告按每个种子先平均 15 种扰动，再计算跨种子的均值和样本标准差。

PEA 直接调用 [官方 ViT 实现](https://github.com/TheMaXiao/PEA_TTA/tree/9c358ba682052862df0739fbbadb71f1b586614d)。每个种子默认用 50,000 张干净训练图估计源统计，随后跨 corruption 复用。适应使用官方两阶段对齐、增强、每次一个增强视图的 microbatch 和熵触发重置。验证保留适应结束时的目标统计与最后一批权重，固定增强视图，禁用验证批的权重估计、目标统计更新与熵重置。这是 512 张适应后冻结验证的协议。PLEAD 使用仓库最终版的默认窗口及校正设置。

时延使用 CUDA 同步后的逐批计时，包含设备传输与方法推理，排除 DataLoader 及离线源统计。逐流 JSON 另存总时间、离线准备或初始化时间、p95 和前向次数。RAM 统计单进程采样峰值，VRAM 使用 PyTorch 已分配显存峰值。模型以 FP32 在正式计时前做一次无状态前向预热。方法间按数据集使用同一块 GPU 串行执行。

## 运行

从仓库根目录在已有的 `df-atta` 环境运行。额外 Python 依赖为 `psutil`。官方 PEA checkout 默认位于 `paper-exp/.upstream/pea`，也可用 `PEA_UPSTREAM` 指定其父目录。脚本会先跑 gaussian_noise 的两种方法 Smoke Test，随后对通过检查的方法运行 45 条正式流。

```bash
bash balanced-512/setup.sh
CUDA_VISIBLE_DEVICES=0 bash balanced-512/run_dataset.sh cifar10-c
CUDA_VISIBLE_DEVICES=0 bash balanced-512/run_dataset.sh cifar100-c
CUDA_VISIBLE_DEVICES=1 bash balanced-512/run_dataset.sh imagenet-c --source-root /path/to/clean/imagenet
python balanced-512/report.py
```

没有干净 ImageNet 时，省略 `--source-root` 可执行 ImageNet-C 的 PLEAD。此时 PEA 留空并记录缺失源数据。提供源数据后可补跑，已有成功记录会跳过。环境变量 `BALANCED_512_OUTPUT` 可指定独立的输出目录，并用 `report.py --output` 汇总该目录。

`results/full/` 保存配置、checkpoint 审计、共享划分索引和哈希、逐流 JSON 以及状态。正式汇总位于 `results/report.md` 与 `results/per_stream.csv`。Smoke Test 位于单独的 `results/smoke/`。

如果相同版本的 checkpoint 未在 Hugging Face 缓存中，可执行
`HF_HOME="${HF_HOME:-$HOME/.cache/neo-huggingface}" python balanced-512/prepare_models.py`。
这个脚本只下载已固定提交的模型文件。评估入口仍从本地缓存严格加载。
